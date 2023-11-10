import torch
from torch.nn import Conv2d,ConvTranspose2d,ModuleList,Identity,LeakyReLU,Sequential
import torch.nn.functional as F
import collections
from utils.uflow_utils import upsample,flow_to_warp
import lightning as pl
import torch.nn as nn
def resample(source, coords):
    """Resample the source image at the passed coordinates.
    Args:
        source: tf.tensor, batch of images to be resampled.
        coords: tf.tensor, batch of coordinates in the image.
    Returns:
        The resampled image.
    Coordinates should be between 0 and size-1. Coordinates outside of this range
    are handled by interpolating with a background image filled with zeros in the
    same way that SAME size convolution works.
    """
    _, _, H, W = source.shape
    # normalize coordinates to [-1 .. 1] range
    coords = coords.clone()
    coords[:, 0, :, :] = 2.0 * coords[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    coords[:, 1, :, :] = 2.0 * coords[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    coords = coords.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(source, coords, align_corners=False)
    return output

def normalize_features(feature_list, normalize, center, moments_across_channels,
                       moments_across_images):
    """Normalizes feature tensors (e.g., before computing the cost volume).

  Args:
    feature_list: list of tf.tensors, each with dimensions [b, h, w, c]
    normalize: bool flag, divide features by their standard deviation
    center: bool flag, subtract feature mean
    moments_across_channels: bool flag, compute mean and std across channels
    moments_across_images: bool flag, compute mean and std across images

  Returns:
    list, normalized feature_list
  """

    # Compute feature statistics.

    statistics = collections.defaultdict(list)
    dims = (-3, -2, -1) if moments_across_channels else (-2, -1)
    for feature_image in feature_list:
        variance,mean = torch.var_mean(input=feature_image, dim=dims, keepdim=True)
        # tf.nn.moments(x=feature_image, axes=axes, keepdims=True)
        statistics['mean'].append(mean)
        statistics['var'].append(variance)

    if moments_across_images:
        statistics['mean'] = ([torch.mean(input=torch.tensor(statistics['mean']))] *
                              len(feature_list))
        statistics['var'] = [torch.mean(input=torch.tensor(statistics['var']))
                             ] * len(feature_list)

    statistics['std'] = [torch.sqrt(v + 1e-16) for v in statistics['var']]

    # Center and normalize features.

    if center:
        feature_list = [
            f - mean for f, mean in zip(feature_list, statistics['mean'])
        ]
    if normalize:
        feature_list = [f / std for f, std in zip(feature_list, statistics['std'])]

    return feature_list
def compute_cost_volume(features1, features2, max_displacement):
    """Compute the cost volume between features1 and features2.

  Displace features2 up to max_displacement in any direction and compute the
  per pixel cost of features1 and the displaced features2.

  Args:
    features1: tf.tensor of shape [b, h, w, c]
    features2: tf.tensor of shape [b, h, w, c]
    max_displacement: int, maximum displacement for cost volume computation.

  Returns:
    tf.tensor of shape [b, h, w, (2 * max_displacement + 1) ** 2] of costs for
    all displacements.
  """

    # Set maximum displacement and compute the number of image shifts.
    _, _,height, width= features1.shape
    if max_displacement <= 0 or max_displacement >= height:
        raise ValueError(f'Max displacement of {max_displacement} is too large.')

    max_disp = max_displacement
    num_shifts = 2 * max_disp + 1

    # Pad features2 and shift it while keeping features1 fixed to compute the
    # cost volume through correlation.

    # Pad features2 such that shifts do not go out of bounds.
    features2_padded = F.pad(
        input=features2,
        pad=(max_disp, max_disp, max_disp, max_disp, 0, 0, 0, 0),
        mode='constant')
    cost_list = []
    for i in range(num_shifts):
        for j in range(num_shifts):
            corr = torch.mean(
                input=features1 *
                      features2_padded[:, :, i:(height + i), j:(width + j)],
                dim=1,
                keepdim=True)
            cost_list.append(corr)
    cost_volume = torch.concat(cost_list, dim=1)  # axis=-1
    return cost_volume




class PWCFlow(pl.LightningModule):
  def __init__(self,
                 leaky_relu_alpha=0.1,
                 dropout_rate=0.25,
                 num_channels_upsampled_context=32,
                 num_levels=5,
                 normalize_before_cost_volume=True,
                 channel_multiplier=1.,
                 use_cost_volume=True,
                 use_feature_warp=True,
                 accumulate_flow=True,
                 use_bfloat16=False,
                 shared_flow_decoder=False):
        super(PWCFlow, self).__init__()
        if use_bfloat16:
            self._dtype_policy = torch.bfloat16
        else:
            self._dtype_policy = torch.float32
        self._leaky_relu_alpha = leaky_relu_alpha
        self._drop_out_rate = dropout_rate
        self._num_context_up_channels = num_channels_upsampled_context
        self._num_levels = num_levels
        self._normalize_before_cost_volume = normalize_before_cost_volume
        self._channel_multiplier = channel_multiplier
        self._use_cost_volume = use_cost_volume
        self._use_feature_warp = use_feature_warp
        self._accumulate_flow = accumulate_flow
        self._shared_flow_decoder = shared_flow_decoder

        self._refine_model = self._build_refinement_model()
        self._flow_layers = ModuleList(self._build_flow_layers())
        if not self._use_cost_volume:
            self._cost_volume_surrogate_convs = ModuleList(self._build_cost_volume_surrogate_convs())
            self.freeze_weight(self._cost_volume_surrogate_convs)
        if self._num_context_up_channels:
            self._context_up_layers = ModuleList(self._build_upsample_layers(out_channel=int(self._num_context_up_channels * self._channel_multiplier)))
            self.freeze_weight(self._context_up_layers)
        if self._shared_flow_decoder:
            self._1x1_shared_decoder = ModuleList(self._build_1x1_shared_decoder())
            self.freeze_weight(self._1x1_shared_decoder)
        self.activation = LeakyReLU(negative_slope=self._leaky_relu_alpha)

        self.freeze_weight(self._flow_layers)
  def forward(self, feature_pyramid1, feature_pyramid2, training=False):
        flow_up = None
        context_up = None
        index = 0
        flows = []
        context = None
        flow = None
        for level, (features1, features2) in reversed(list(enumerate(zip(feature_pyramid1, feature_pyramid2)))[1:]):
            if self._shared_flow_decoder and flow_up is None:
                batch_size, _, height, width = features1.shape
                flow_up = torch.zeros(
                    (batch_size, 2, height, width))
                if self._num_context_up_channels:
                    num_channels = int(self._num_context_up_channels *
                                       self._channel_multiplier)
                    context_up = torch.zeros(
                        (batch_size, num_channels, height, width))

            if flow_up is None or not self._use_feature_warp:
                warped2 = features2
            else:
                warp_up = flow_to_warp(flow_up)
                warped2 = resample(features2, warp_up)
            features1_normalized, warped2_normalized = normalize_features(
                [features1, warped2],
                normalize=self._normalize_before_cost_volume,
                center=self._normalize_before_cost_volume,
                moments_across_channels=True,
                moments_across_images=True)

            # print('feature1_normalized',features1_normalized.shape)
            # print('warp2_normalized',warped2_normalized.shape)
            if self._use_cost_volume:  # if self._use_cost_volume:
                cost_volume = compute_cost_volume(
                    features1_normalized, warped2_normalized, max_displacement=4)
            else:
                concat_features = torch.cat((features1_normalized, warped2_normalized), dim=1)
                # Concatenate(axis=-1)([features1_normalized, warped2_normalized])
                cost_volume = self._cost_volume_surrogate_convs[level](concat_features)

            cost_volume = self.activation(cost_volume)

            if self._shared_flow_decoder:
                # This will ensure to work for arbitrary feature sizes per level.
                features1 = self._1x1_shared_decoder[level](features1)

            if flow_up is None:
                x_in = torch.cat((cost_volume, features1), dim=1)  # Concatenate(axis=-1)([cost_volume, features1])
            else:
                if context_up is None:
                    x_in = torch.cat((flow_up, cost_volume, features1),
                                     dim=1)  # Concatenate(axis=-1)([flow_up, cost_volume, features1])
                else:

                    x_in = torch.cat((context_up, flow_up, cost_volume, features1), dim=1)
                    # Concatenate(axis=-1)([context_up, flow_up, cost_volume, features1])

                # Use dense-net connections.
            x_out = None
            # reuse the same flow decoder on all levels
            if self._shared_flow_decoder:
                # reuse the same flow decoder on all levels
                flow_layers = self._flow_layers
            else:
                flow_layers = self._flow_layers[level]

            for layer in flow_layers[:-1]:
                x_out = layer(x_in)
                x_in = torch.cat((x_in, x_out), dim=1)  # Concatenate(axis=-1)([x_in, x_out])

            context = x_out

            flow = flow_layers[-1](context)

            if (training and self._drop_out_rate):
                maybe_dropout = (torch.rand([]) > self._drop_out_rate).type(self._dtype_policy)
                '''
        tf.cast(
            tf.math.greater(tf.random.uniform([]), self._drop_out_rate),
            tf.bfloat16 if self._use_bfloat16 else tf.float32)
        '''
                context =context * maybe_dropout
                flow = flow * maybe_dropout

            if flow_up is not None and self._accumulate_flow:
                flow = flow + flow_up

            # Upsample flow for the next lower level.
            flow_up = upsample(flow, is_flow=True)
            if self._num_context_up_channels:
                context_up = self._context_up_layers[level](context)
            index += 1

            flows.insert(0, flow)


        cf = torch.concat([context, flow],dim=1)  # concat the context and flow
        refinement = self._refine_model(cf)

        if (training and self._drop_out_rate):
            refinement = refinement * (torch.rand([]) > self._drop_out_rate)
        refined_flow = flow + refinement
        flows[0] = refined_flow

        return [flow.type(torch.float32) for flow in flows]

  def freeze_weight(self,model):
      for param in model.parameters():
          param.requires_grad = False

  def _build_refinement_model(self):
        """Build model for flow refinement using dilated convolutions."""
        in_channel = 34
        layers = []
        for c, d in [(128, 1), (128, 2), (128, 4), (96, 8), (64, 16), (32, 1)]:
            layers.append(Sequential(
                Conv2d(
                    in_channels=in_channel,
                    out_channels=int(c * self._channel_multiplier),
                    kernel_size=(3, 3),
                    stride=1,
                    padding=(d*(3-1)//2,d*(3-1)//2), # dilation * (kernal_size - 1) //2
                    dilation=d),
                LeakyReLU(negative_slope=self._leaky_relu_alpha))
            )
            in_channel = int(c * self._channel_multiplier)
        layers.append(
                Conv2d(
                in_channels=in_channel,
                out_channels=2,
                kernel_size=(3, 3),
                stride=1,
                padding=1))
        return Sequential(*layers)


  def _build_cost_volume_surrogate_convs(self):
      layers = []
      for _ in range(self._num_levels):
          layers.append(
                  Conv2d(
                  in_channels=32,
                  out_channels=int(64 * self._channel_multiplier),
                  kernel_size=(4, 4),
                  # P = ((S-1)*W-S+F)/2, with F = filter size, S = stride, W = input size
                  stride=(1, 1)
              )
          )
      
      return layers

  def _build_upsample_layers(self, out_channel):
      """Build layers for upsampling via deconvolution."""
      layers = []
      for unused_level in range(self._num_levels):
      #for unused_level in range(self._num_levels):
          layers.append(
                ConvTranspose2d(
                 in_channels=32,
                  out_channels=out_channel,
                  kernel_size=(4, 4),
                  stride=2,
                  # padding='same',
                  padding=1))
      return layers

  def _build_flow_layers(self):
      """Build layers for flow estimation."""
      # Empty list of layers level 0 because flow is only estimated at levels > 0.
      result = [Identity()]
      in_channel = 0
      channel_last_level = [113,241,369,465,529]
      channel_rest_level = [147,275,403,499,563]
      for i in range(1, self._num_levels-1):
          layer = []
          for o,i in zip([128, 128, 96, 64, 32],channel_rest_level):
              layer.append(Sequential(
                      Conv2d(
                          in_channels=i,
                          out_channels=int(o * self._channel_multiplier),
                          kernel_size=(3, 3),
                          stride=1,
                          padding=1),
                      LeakyReLU(
                          negative_slope=self._leaky_relu_alpha)
              )
              )
              in_channel = int(o * self._channel_multiplier)
          layer.append(
              Conv2d(
                  in_channels=in_channel,
                  out_channels=2,
                  kernel_size=(3, 3),
                  stride=1,
                  padding=1
              ))
          L = Sequential(*layer)
          if self._shared_flow_decoder:
              return L
          result.append(L)
      layer = []
      for o, i in zip([128, 128, 96, 64, 32], channel_last_level):
          layer.append(Sequential(
              Conv2d(
                  in_channels=i,
                  out_channels=int(o * self._channel_multiplier),
                  kernel_size=(3, 3),
                  stride=1,
                  padding=1),
              LeakyReLU(
                  negative_slope=self._leaky_relu_alpha)
          )
          )
          in_channel = int(o * self._channel_multiplier)

      layer.append(
          Conv2d(
              in_channels=in_channel,
              out_channels=2,
              kernel_size=(3, 3),
              stride=1,
              padding=1
          ))
      L = Sequential(*layer)
      if self._shared_flow_decoder:
          return L
      result.append(L)

      return result


  def _build_1x1_shared_decoder(self):
      """Build layers for flow estimation."""
      # Empty list of layers level 0 because flow is only estimated at levels > 0.
      result = [Identity()]
      for _ in range(1, self._num_levels):
            result.append(Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(1, 1),
                stride=1))
      return result


if __name__ == '__main__':
    uflow = PWCFlow()
    for param in uflow.parameters():
        print(param)