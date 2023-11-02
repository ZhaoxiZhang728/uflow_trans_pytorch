#%%

#%%
# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""UFlow utils.

This library contains the various util functions used in UFlow.
"""
'''
For tensor flow, the default vecoter [b,c,h,w]
'''
import time

#import tensorflow as tf
from utils import uflow_plotting
#from utils.uflow_resampler import resampler
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TTF
from func_py.tf_to_py_warping import unsorted_segment_sum
def flow_to_warp(flow):
  """Compute the warp from the flow field.

  Args:
    flow: tf.tensor representing optical flow. now torch.tensor

  Returns:
    The warp, i.e. the endpoints of the estimated flow.
  """
  #print(flow.shape)
  # Construct a grid of the image coordinates.
  device = flow.device
  height, width = flow.shape[-2:]
  i_grid, j_grid = torch.meshgrid(
      torch.linspace(0.0, height - 1.0, int(height),device=device),
      torch.linspace(0.0, width - 1.0, int(width),device=device),
      indexing='ij')
  grid = torch.stack([i_grid, j_grid], dim=0)
  #print("grid_shape",i_grid.shape)
  # Potentially add batch dimension to match the shape of flow.
  if len(flow.shape) == 4:
    grid = grid[None]
  # Add the flow field to the image grid.
  if flow.dtype != grid.dtype:
    grid = grid.type(flow.dtype)

  warp = flow + grid
  return warp


def mask_invalid(coords):
  """Mask coordinates outside of the image.

  Valid = 1, invalid = 0.

  Args:
    coords: a 4D float tensor of image coordinates.

  Returns:
    The mask showing which coordinates are valid.
  """
  coords_rank = len(coords.shape)
  if coords_rank != 4:
    raise NotImplementedError()
  max_height = float(coords.shape[-2] - 1)
  max_width = float(coords.shape[-1] - 1)
  mask = torch.logical_and(
      torch.logical_and(coords[:, 0, :, :] >= 0.0,
                        coords[:, 0, :, :] <= max_height),
      torch.logical_and(coords[:, 1, :, :] >= 0.0,
                        coords[:, 1, :, :] <= max_width))
  mask = mask[:, None, :, :]
  return mask.type(torch.float32)

'''
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

  # Wrap this function because it uses a different order of height/width dims.
  orig_source_dtype = source.dtype
  if source.dtype != torch.float32:
    source = source.type(torch.float32)
  if coords.dtype != torch.float32:
    coords = coords.type(torch.float32)
  coords_rank = len(coords.shape)
  if coords_rank == 4:
    output = resampler(source, coords.flip(1))
    if orig_source_dtype != source.dtype:
      return output.type(orig_source_dtype)
    return output
  else:
    raise NotImplementedError()
'''
def reflect(x, minx, maxx):
    """ Reflects an array around two points making a triangular waveform that ramps up
    and down,  allowing for pad lengths greater than the input length """
    rng = maxx - minx
    double_rng = 2*rng
    mod = torch.fmod(x - minx, double_rng)
    normed_mod = torch.where(mod < 0, mod+double_rng, mod)
    out = torch.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx

    output = out.clone().detach()
    output = output.type(x.dtype)
    return output

def batch_symmetric_padding_pytorch(im: torch.Tensor, padding):
    '''

    :param im: batch image with channels
    :param padding: [[],[],[],[]]
    :return: after padding image
    '''
    b,c, h, w = im.shape
    left, right = padding[-1] # the final list of padding is for left and right
    top, bottom = padding[-2] # the final second list of padding is for top and right

    c_top, c_bottom = padding[-3] # Channel
    b_top,b_buttom =padding[-4] # batch
    x_idx = torch.arange(-left, w+right)
    y_idx = torch.arange(-top, h+bottom)
    c_idx = torch.arange(-c_top,c + c_bottom)
    b_idx = torch.arange(-b_top,b+b_buttom)
    #print(x_idx)
    #print(y_idx)
    #print(c_idx)
    #print(b_idx)
    x_pad = reflect(x_idx, -0.5, w-0.5)
    y_pad = reflect(y_idx, -0.5, h-0.5)
    c_pad = reflect(c_idx, -0.5, c-0.5)
    b_pad = reflect(b_idx, -0.5, b-0.5)

    xx, yy = torch.meshgrid(x_pad, y_pad,indexing='xy')
    cc, bb = torch.meshgrid(c_pad, b_pad,indexing='xy')
    result = im[..., yy, xx].clone().detach()# get the matrix first and repulicate

    output = result[bb,cc,...]

    return output
def compute_range_map(flow,
                      downsampling_factor=1,
                      reduce_downsampling_bias=True,
                      resize_output=True):
  """Count how often each coordinate is sampled.

  Counts are assigned to the integer coordinates around the sampled coordinates
  using weights from bilinear interpolation.

  Args:
    flow: A float tensor of shape (batch size x height x width x 2) that
      represents a dense flow field.
    downsampling_factor: An integer, by which factor to downsample the output
      resolution relative to the input resolution. Downsampling increases the
      bin size but decreases the resolution of the output. The output is
      normalized such that zero flow input will produce a constant ones output.
    reduce_downsampling_bias: A boolean, whether to reduce the downsampling bias
      near the image boundaries by padding the flow field.
    resize_output: A boolean, whether to resize the output ot the input
      resolution.

  Returns:
    A float tensor of shape [batch_size, height, width, 1] that denotes how
    often each pixel is sampled.
  """
  device = flow.device
  # Get input shape.
  input_shape = list(flow.shape)
  if len(input_shape) != 4:
    raise NotImplementedError()
  batch_size,_,input_height, input_width = input_shape

  flow_height = input_height
  flow_width = input_width

  # Apply downsampling (and move the coordinate frame appropriately).
  output_height = input_height // downsampling_factor
  output_width = input_width // downsampling_factor
  if downsampling_factor > 1:
    # Reduce the bias that comes from downsampling, where pixels at the edge
    # will get lower counts that pixels in the middle of the image, by padding
    # the flow field.
    if reduce_downsampling_bias:
      p = downsampling_factor // 2
      flow_height += 2 * p
      flow_width += 2 * p
      # Apply padding in multiple steps to padd with the values on the edge.
      for _ in range(p):
        flow = batch_symmetric_padding_pytorch( # tf.pad(...,mode = 'SYMMETRIC')
            im=flow,
            padding=[[0, 0], [0, 0],[1, 1], [1, 1]])
      coords = flow_to_warp(flow) - p
    # Update the coordinate frame to the downsampled one.
      coords = (coords + (1 - downsampling_factor) * 0.5) / downsampling_factor
  elif downsampling_factor == 1:
    coords = flow_to_warp(flow)
  else:
    raise ValueError('downsampling_factor must be an integer >= 1.')

  # Split coordinates into an integer part and a float offset for interpolation.
  coords_floor = torch.floor(input=coords)
  coords_offset = coords - coords_floor
  coords_floor = coords_floor.type(torch.int32)# original is tf.cast(..,'int32')

  # Define a batch offset for flattened indexes into all pixels.
  batch_range = torch.reshape(torch.arange(batch_size,device=device), [batch_size, 1, 1])
  idx_batch_offset = torch.tile(
      batch_range, [1, flow_height, flow_width]) * output_height * output_width

  # Flatten everything.
  coords_floor_flattened = torch.reshape(coords_floor, [-1, 2])
  coords_offset_flattened = torch.reshape(coords_offset, [-1, 2])
  idx_batch_offset_flattened = torch.reshape(idx_batch_offset, [-1])

  # Initialize results.
  idxs_list = []
  weights_list = []

  # Loop over differences di and dj to the four neighboring pixels.
  for di in range(2):
    for dj in range(2):

      # Compute the neighboring pixel coordinates.
      idxs_i = coords_floor_flattened[:, 0] + di
      idxs_j = coords_floor_flattened[:, 1] + dj
      # Compute the flat index into all pixels.
      idxs = idx_batch_offset_flattened + idxs_i * output_width + idxs_j

      # Only count valid pixels.
      mask = torch.reshape(torch.nonzero(torch.logical_and(
      torch.logical_and(idxs_i >= 0, idxs_i < output_height),
      torch.logical_and(idxs_j >= 0, idxs_j < output_width))), [-1]).type(torch.int64)
      '''
      torch.reshape(
          torch.where( #tf.compat.v1.where not sure the equivalent function in pytorch
              condition=torch.logical_and(
                  torch.logical_and(idxs_i >= 0, idxs_i < output_height),
                  torch.logical_and(idxs_j >= 0, idxs_j < output_width))), [-1])
      '''
      valid_idxs = idxs[mask]#torch.gather(input=idxs,index=mask.type(torch.int64),dim=0)
      valid_offsets = coords_offset_flattened[mask]#torch.gather(input=coords_offset_flattened, index=mask.type(torch.int64),dim=0)

      # Compute weights according to bilinear interpolation.
      weights_i = (1. - di) - (-1)**di * valid_offsets[:, 0]
      weights_j = (1. - dj) - (-1)**dj * valid_offsets[:, 1]
      weights = weights_i * weights_j

      # Append indices and weights to the corresponding list.
      idxs_list.append(valid_idxs)
      weights_list.append(weights)

  # Concatenate everything.
  idxs = torch.concat(idxs_list, dim=0)
  weights = torch.concat(weights_list, dim=0)

  # Sum up weights for each pixel and reshape the result.
  counts = unsorted_segment_sum(
      weights, idxs, batch_size * output_height * output_width)

  count_image = torch.reshape(counts, [batch_size,1,output_height, output_width])

  if downsampling_factor > 1:
    # Normalize the count image so that downsampling does not affect the counts.
    count_image /= downsampling_factor**2
    if resize_output:
      count_image = resize(
          count_image, input_height, input_width, is_flow=False)

  return count_image
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

def compute_warps_and_occlusion(flows,
                                occlusion_estimation,
                                occ_weights=None,
                                occ_thresholds=None,
                                occ_clip_max=None,
                                occlusions_are_zeros=True,
                                occ_active=None):
  """Compute warps, valid warp masks, advection maps, and occlusion masks."""

  if occ_clip_max is not None:
    for key in occ_clip_max:
      if key not in ['forward_collision', 'fb_abs']:
        raise ValueError('occ_clip_max for this key is not supported')

  warps = dict()
  range_maps_high_res = dict()
  range_maps_low_res = dict()
  occlusion_logits = dict()
  occlusion_scores = dict()
  occlusion_masks = dict()
  valid_warp_masks = dict()
  fb_sq_diff = dict()
  fb_sum_sq = dict()

  for key in flows:

    i,j,t = key
    rev_key = (j,i,t)

    warps[key] = []
    range_maps_high_res[key] = []
    range_maps_low_res[rev_key] = []
    occlusion_masks[key] = []
    valid_warp_masks[key] = []
    fb_sq_diff[key] = []
    fb_sum_sq[key] = []


    for level in range(min(3, len(flows[key]))):

      flow_ij = flows[key][level]
      flow_ji = flows[rev_key][level]

      # Compute warps (coordinates) and a mask for which coordinates are valid.
      warps[key].append(flow_to_warp(flow_ij))
      valid_warp_masks[key].append(mask_invalid(warps[key][level]))

      # Compare forward and backward flow.
      flow_ji_in_i = resample(flow_ji, warps[key][level])
      #print(flow_ij.shape,flow_ji_in_i.shape)
      fb_sq_diff[key].append(
          torch.sum(input=(flow_ij + flow_ji_in_i)**2,dim=1,keepdim=True)
          )#tf.reduce_sum(input_tensor=(flow_ij + flow_ji_in_i)**2, axis=-1, keepdims=True)
      fb_sum_sq[key].append(
          torch.sum(input=(flow_ij**2 + flow_ji_in_i**2),dim=1,keepdim=True)
      )# tf.reduce_sum(input_tensor=(flow_ij**2 + flow_ji_in_i**2),axis=-1,keepdims=True)

      if level != 0:
        continue

      # This initializations avoids problems in tensorflow (likely AutoGraph)
      occlusion_mask = torch.zeros_like(flow_ij[:, :1, Ellipsis], dtype=torch.float32)
      occlusion_scores['forward_collision'] = torch.zeros_like(flow_ij[:, :1, Ellipsis],dtype=torch.float32)
      #Before: torch.zeros_like(flow_ij[Ellipsis, :1], dtype=torch.float32)
      #print(occlusion_scores['forward_collision'].shape)
      occlusion_scores['backward_zero'] = torch.zeros_like(
          flow_ij[:, :1, Ellipsis], dtype=torch.float32)
      occlusion_scores['fb_abs'] = torch.zeros_like(
          flow_ij[:, :1, Ellipsis], dtype=torch.float32)

      if occlusion_estimation == 'none' or (
          occ_active is not None and not occ_active[occlusion_estimation]):
        occlusion_mask = torch.zeros_like(flow_ij[:, :1, Ellipsis], dtype=torch.float32)

      elif occlusion_estimation == 'brox':
        occlusion_mask = (fb_sq_diff[key][level] > 0.01 * fb_sum_sq[key][level] + 0.5).type(torch.float32)
        # ... > ...
      elif occlusion_estimation == 'fb_abs':
        occlusion_mask = (fb_sq_diff[key][level]**0.5 > 1.5).type(torch.float32)


      elif occlusion_estimation == 'wang':
        range_maps_low_res[rev_key].append(
            compute_range_map(
                flow_ji,
                downsampling_factor=1,
                reduce_downsampling_bias=False,
                resize_output=False))
        # Invert so that low values correspond to probable occlusions,
        # range [0, 1].
        occlusion_mask = (
            1. - torch.clamp(range_maps_low_res[rev_key][level], 0., 1.))

      elif occlusion_estimation == 'wang4':
        range_maps_low_res[rev_key].append(
            compute_range_map(
                flow_ji,
                downsampling_factor=4,
                reduce_downsampling_bias=True,
                resize_output=True))
        # Invert so that low values correspond to probable occlusions,
        # range [0, 1].
        occlusion_mask = (
            1. - torch.clamp(range_maps_low_res[rev_key][level], 0., 1.))

      elif occlusion_estimation == 'wangthres':
        range_maps_low_res[rev_key].append(
            compute_range_map(
                flow_ji,
                downsampling_factor=1,
                reduce_downsampling_bias=True,
                resize_output=True))
        # Invert so that low values correspond to probable occlusions,
        # range [0, 1].
        occlusion_mask = range_maps_low_res[rev_key][level] < 0.75
        occlusion_mask = occlusion_mask.type(torch.float32)

      elif occlusion_estimation == 'wang4thres':
        range_maps_low_res[rev_key].append(
            compute_range_map(
                flow_ji,
                downsampling_factor=4,
                reduce_downsampling_bias=True,
                resize_output=True))
        # Invert so that low values correspond to probable occlusions,
        # range [0, 1].
        occlusion_mask = range_maps_low_res[rev_key][level] < 0.75
        occlusion_mask = occlusion_mask.type(torch.float32)

      elif occlusion_estimation == 'uflow':
        # Compute occlusion from the range map of the forward flow, projected
        # back into the frame of image i. The idea is if many flow vectors point
        # to the same pixel, those are likely occluded.
        if 'forward_collision' in occ_weights and (
            occ_active is None or occ_active['forward_collision']):
          range_maps_high_res[key].append(
              compute_range_map(
                  flow_ij,
                  downsampling_factor=1,
                  reduce_downsampling_bias=True,
                  resize_output=True))
          fwd_range_map_in_i = resample(range_maps_high_res[key][level],
                                        warps[key][level])
          # Rescale to [0, max-1].
          occlusion_scores['forward_collision'] = torch.clamp(
              fwd_range_map_in_i, 1., occ_clip_max['forward_collision']) - 1.0

        # Compute occlusion from the range map of the backward flow, which is
        # already computed in frame i. Pixels that no flow vector points to are
        # likely occluded.
        if 'backward_zero' in occ_weights and (occ_active is None or
                                               occ_active['backward_zero']):
          range_maps_low_res[rev_key].append(
              compute_range_map(
                  flow_ji,
                  downsampling_factor=4,
                  reduce_downsampling_bias=True,
                  resize_output=True))
          # Invert so that low values correspond to probable occlusions,
          # range [0, 1].
          occlusion_scores['backward_zero'] = (
              1. - torch.clamp(range_maps_low_res[rev_key][level], 0., 1.))

        # Compute occlusion from forward-backward consistency. If the flow
        # vectors are inconsistent, this means that they are either wrong or
        # occluded.
        if 'fb_abs' in occ_weights and (occ_active is None or
                                        occ_active['fb_abs']):
          # Clip to [0, max].
          occlusion_scores['fb_abs'] = torch.clamp(
              fb_sq_diff[key][level]**0.5, 0.0, occ_clip_max['fb_abs'])

        occlusion_logits = torch.zeros_like(flow_ij[:,:1,Ellipsis], dtype=torch.float32)
        for k, v in occlusion_scores.items():
          occlusion_logits += (v - occ_thresholds[k]) * occ_weights[k]
        occlusion_mask = torch.sigmoid(occlusion_logits)
      else:
        raise ValueError('Unknown value for occlusion_estimation:',
                         occlusion_estimation)

      occlusion_masks[key].append(
          1. - occlusion_mask if occlusions_are_zeros else occlusion_mask)

  return warps, valid_warp_masks, range_maps_low_res, occlusion_masks, fb_sq_diff, fb_sum_sq

def apply_warps_stop_grad(sources, warps, level):
  """Apply all warps on the correct sources."""

  warped = dict()
  for (i, j, t) in warps:
    # Only propagate gradient through the warp, not through the source.
    warped[(i, j, t)] = resample(
        sources[j].clone().detach(), warps[(i, j, t)][level])

  return warped


def upsample(img, is_flow):
  """Double resolution of an image or flow field.

  Args:
    img: tf.tensor, image or flow field to be resized
    is_flow: bool, flag for scaling flow accordingly

  Returns:
    Resized and potentially scaled image or flow field.
  """
  _, _, height, width = list(img.shape)
  orig_dtype = img.dtype
  if orig_dtype != torch.float32:
    img = img.type(torch.float32)
  img_resized = torchvision.transforms.Resize(size=(int(height * 2),int(width * 2)))(img)

      #tf.compat.v2.image.resize(img,(int(height * 2), int(width * 2)))
  if is_flow:
    # Scale flow values to be consistent with the new image size.
    img_resized *= 2
  if img_resized.dtype != orig_dtype:
    return img_resized.type(orig_dtype)
  return img_resized


def downsample(img, is_flow):
  """Halve the resolution of an image or flow field.

  Args:
    img: tf.tensor, image or flow field to be resized
    is_flow: bool, flag for scaling flow accordingly

  Returns:
    Resized and potentially scaled image or flow field.
  """
  _, _,height, width = list(img.shape)
  img_resized = torchvision.transforms.Resize(size=(int(height /2),int(width /2)))(img)

      #tf.compat.v2.image.resize(img,(int(height / 2), int(width / 2)))
  if is_flow:
    # Scale flow values to be consistent with the new image size.
    img_resized /= 2
  return img_resized

def reciprocal_no_nan_pytorch(inputs):
    x = torch.reciprocal(inputs)
    result = torch.nan_to_num(input=x, nan=0,posinf=0,neginf=0)

    return result


#@tf.function
def resize(img, height, width, is_flow, mask=None):
  """Resize an image or flow field to a new resolution.

  In case a mask (per pixel {0,1} flag) is passed a weighted resizing is
  performed to account for missing flow entries in the sparse flow field. The
  weighting is based on the resized mask, which determines the 'amount of valid
  flow vectors' that contributed to each individual resized flow vector. Hence,
  multiplying by the reciprocal cancels out the effect of considering non valid
  flow vectors.

  Args:
    img: tf.tensor, image or flow field to be resized of shape [b, h, w, c]
    height: int, heigh of new resolution
    width: int, width of new resolution
    is_flow: bool, flag for scaling flow accordingly
    mask: tf.tensor, mask (optional) per pixel {0,1} flag

  Returns:
    Resized and potentially scaled image or flow field (and mask).
  """

  def _resize(img, mask=None):
    orig_height, orig_width = img.shape[-2:]

    #orig_height = img.shape[1]
    #orig_width = img.shape[2]

    if orig_height == height and orig_width == width:
      # early return if no resizing is required
      if mask is not None:
        return img, mask
      else:
        return img

    if mask is not None:
      # multiply with mask, to ensure non-valid locations are zero
      img = torch.mul(img,mask)#tf.math.multiply(img, mask)
      # resize image

      img_resized = torchvision.transforms.Resize(size=(int(height),int(width)),antialias=True)(img)
          #tf.compat.v2.image.resize(img, (int(height), int(width)), antialias=True)
      # resize mask (will serve as normalization weights)
      mask_resized = torchvision.transforms.Resize(size=(int(height),int(width)),antialias=True)(mask)
      # normalize sparse flow field and mask
      #print(mask_resized)
      img_resized = torch.mul(img_resized,reciprocal_no_nan_pytorch(mask_resized))
      #print(reciprocal_no_nan_pytorch(mask_resized))
      #tf.math.multiply(img_resized,tf.math.reciprocal_no_nan(mask_resized))
      mask_resized = torch.mul(mask_resized,reciprocal_no_nan_pytorch(mask_resized))
      #tf.math.multiply(mask_resized,tf.math.reciprocal_no_nan(mask_resized))
    else:
      # normal resize without anti-alaising
      img_resized = torchvision.transforms.Resize(size=(int(height),int(width)))(img)
      #tf.compat.v2.image.resize(img, (int(height), int(width)))

    if is_flow:
      # If image is a flow image, scale flow values to be consistent with the
      # new image size.
      scaling = torch.reshape(torch.tensor([
          float(height) / orig_height,
          float(width) / orig_width
      ],dtype = torch.float32), [1, 2, 1, 1]) # original shape is [1,1,1,2]
      #print("scaling :",scaling)
      img_resized *= scaling

      #print("img_resized", img_resized.shape)

      '''
      old code:
      scaling = torch.reshape([
          float(height) / orig_height.type(torch.float32),
          float(width) / orig_width.type(torch.float32)
      ], [1, 1, 1, 2])
      img_resized *= scaling
      '''
    if mask is not None:
      return img_resized, mask_resized
    return img_resized

  # Apply resizing at the right shape.
  shape = list(img.shape)
  if len(shape) == 3:
    if mask is not None:
      img_resized, mask_resized = _resize(img[None], mask[None])
      return img_resized[0], mask_resized[0]
    else:
      return _resize(img[None])[0]
  elif len(shape) == 4:
    # Input at the right shape.
    return _resize(img, mask)
  elif len(shape) > 4:
    # Reshape input to [b, h, w, c], resize and reshape back.
    # for pytorch the input should be [b,c,h,w]
    img_flattened = torch.reshape(img, [-1] + shape[-3:])
    if mask is not None:
      mask_flattened = torch.reshape(mask, [-1] + shape[-3:])
      img_resized, mask_resized = _resize(img_flattened, mask_flattened)
    else:
      img_resized = _resize(img_flattened)
    # There appears to be some bug in tf2 tf.function
    # that fails to capture the value of height / width inside the closure,
    # leading the height / width undefined here. Call set_shape to make it
    # defined again.
    img_resized.reshape(
        (img_resized.shape[0],img_resized.shape[1], height, width))
    result_img = torch.reshape(img_resized, shape[:-3] + img_resized.shape[-3:])
    if mask is not None:
      mask_resized.set_shape(
          (mask_resized.shape[0], mask_resized.shape[1],height, width))
      result_mask = torch.reshape(mask_resized,
                               shape[:-3] + mask_resized.shape[-3:])
      return result_img, result_mask
    return result_img
  else:
    raise ValueError('Cannot resize an image of shape', shape)


def random_subseq(sequence, subseq_len):
  """Select a random subsequence of a given length."""
  seq_len = sequence.shape[0]
  minval=0
  maxval=seq_len - subseq_len + 1
  start_index = ((maxval-minval)*torch.rand([]) + minval).type(torch.int32)
  #tf.random.uniform([],minval=0,maxval=seq_len - subseq_len + 1,dtype=tf.int32)
  subseq = sequence[start_index:start_index + subseq_len]
  return subseq


def normalize_for_feature_metric_loss(features):
  """Normalize features for the feature-metric loss."""
  normalized_features = dict()
  for key, feature_map in features.items():
    # Normalize feature channels to have the same absolute activations.
    norm_feature_map = feature_map / ((torch.sum(input=abs(feature_map),dim=[0, 2, 3], keepdim=True)) +1e-16)
        #tf.reduce_sum(
            #input_tensor=abs(feature_map), axis=[0, 1, 2], keepdims=True) +
        #1e-16)
    # Normalize every pixel feature across all channels to have mean 1.
    norm_feature_map /= ((torch.sum(input=abs(feature_map),dim=[0, 2, 3], keepdim=True))+1e-16)
    #(
        #tf.reduce_sum(
            #input_tensor=abs(norm_feature_map), axis=[-1], keepdims=True) +
        #1e-16)
    normalized_features[key] = norm_feature_map
  return normalized_features






def image_averages(image_batch):
  image_batch_ah = (image_batch[:, :, 1:] + image_batch[:, :, -1]) / 2.
  image_batch_aw = (image_batch[:, :, :,1:] + image_batch[:, :, :, -1]) / 2
  return image_batch_ah, image_batch_aw






def compute_features_and_flow(
    feature_model,
    flow_model,
    batch,
    batch_without_aug,
    training,
    build_selfsup_transformations=None,
    teacher_feature_model=None,
    teacher_flow_model=None,
    teacher_image_version='original',
):
  """Compute features and flow for an image batch.

  Args:
    feature_model: A model to compute features for flow.
    flow_model: A model to compute flow.
    batch: A tf.tensor of shape [b, seq, h, w, c] holding a batch of triplets.
    batch_without_aug: Batch without photometric augmentation
    training: bool that tells the model to use training or inference code.
    build_selfsup_transformations: A function which, when called with images
      and flows, populates the images and flows dictionary with student images
      and modified teacher flows corresponding to the student images.
    teacher_feature_model: None or instance of of feature model. If None, will
      not compute features and images for teacher distillation.
    teacher_flow_model: None or instance of flow model. If None, will not
      compute features and images for teacher distillation.
    teacher_image_version: str, either 'original' or 'augmented'

  Returns:
    A tuple consisting of the images, the extracted features, the estimated
    flows, and the upsampled refined flows.
  """

  images = dict()
  flows = dict()
  features = dict()

  seq_len = int(batch.shape[1])

  perform_selfsup = (
      training and teacher_feature_model is not None and
      teacher_flow_model is not None and
      build_selfsup_transformations is not None)
  if perform_selfsup:
    selfsup_transform_fns = build_selfsup_transformations()
  else:
    selfsup_transform_fns = None

  for i in range(seq_len):
    # Populate teacher images with native, unmodified images.
    images[(i, 'original')] = batch_without_aug[:, i]
    images[(i, 'augmented')] = batch[:, i]
    if perform_selfsup:
      images[(i, 'transformed')] = selfsup_transform_fns[0](
          images[(i, 'augmented')], i_or_ij=i, is_flow=False)

  for key, image in images.items():
    i, image_version = key
    # if perform_selfsup and image_version == 'original':
    if perform_selfsup and image_version == teacher_image_version:
      features[(i, 'original-teacher')] = teacher_feature_model(
          image, split_features_by_sample=False, training=False)

    features[(i, image_version + '-student')] = feature_model(
        image, split_features_by_sample=False, training=training)

  # Only use original images and features computed on those for computing
  # photometric losses down the road.
  images = {i: images[(i, 'original')] for i in range(seq_len)}

  # Compute flow for all pairs of consecutive images that have the same (or no)
  # transformation applied to them, i.e. that have the same t.
  # pylint:disable=dict-iter-missing-items
  for (i, ti) in features:
    for (j, tj) in features:
      if (i + 1 == j or i - 1 == j) and ti == tj:
        t = ti
        key = (i, j, t)
        # No need to compute the flow for student applied to the original
        # image. We just need the features from that for the photometric loss.
        if t in ['augmented-student', 'transformed-student']:
          # Compute flow from i to j, defined in image i.
          flow = flow_model(
              features[(i, t)], features[(j, t)], training=training)

        elif t in ['original-teacher']:
          flow = teacher_flow_model(
              features[(i, t)], features[(j, t)], training=False)
        else:
          continue

        # Keep flows at levels 0-2.
        flow_level2 = flow[0]
        flow_level1 = upsample(flow_level2, is_flow=True)
        flow_level0 = upsample(flow_level1, is_flow=True)
        flows[key] = [flow_level0, flow_level1, flow_level2]

  return flows, selfsup_transform_fns


def compute_flow_for_supervised_loss(feature_model, flow_model, batch,
                                     training):
  """Compute features and flow for an image batch.

  Args:
    feature_model: A model to compute features for flow.
    flow_model: A model to compute flow.
    batch: A tf.tensor of shape [b, seq, h, w, c] holding a batch of triplets.
    training: bool that tells the model to use training or inference code.

  Returns:
    A tuple consisting of the images, the extracted features, the estimated
    flows, and the upsampled refined flows.
  """

  flows = dict()

  image_0 = batch[:, 0]
  image_1 = batch[:, 1]

  features_0 = feature_model(
      image_0, split_features_by_sample=False, training=training)
  features_1 = feature_model(
      image_1, split_features_by_sample=False, training=training)

  flow = flow_model(features_0, features_1, training=training)
  flow_level2 = flow[0]
  flow_level1 = upsample(flow_level2, is_flow=True)
  flow_level0 = upsample(flow_level1, is_flow=True)
  flows[(0, 1, 'augmented')] = [flow_level0, flow_level1, flow_level2]

  return flows


def random_crop(batch, max_offset_height=32, max_offset_width=32):
  """Randomly crop a batch of images.

  Args:
    batch: a 4-D tensor of shape [batch_size, height, width, num_channels].
    max_offset_height: an int, the maximum vertical coordinate of the top left
      corner of the cropped result.
    max_offset_width: an int, the maximum horizontal coordinate of the top left
      corner of the cropped result.

  Returns:
    a pair of 1) the cropped images in form of a tensor of shape
    [batch_size, height-max_offset, width-max_offset, num_channels],
    2) an offset tensor of shape [batch_size, 2] for height and width offsets.
  """

  # Compute current shapes and target shapes of the crop.
  batch_size, num_channels, height, width= batch.shape
  target_height = height - max_offset_height
  target_width = width - max_offset_width

  # Randomly sample offsets.
  offsets_height = ((max_offset_height + 1) * torch.rand([batch_size])).type(torch.int32)
  '''tf.random.uniform([batch_size],
                                     maxval=max_offset_height + 1,
                                     dtype=tf.int32)'''
  offsets_width = ((max_offset_width + 1) * torch.rand([batch_size])).type(torch.int32)
  '''tf.random.uniform([batch_size],
                                    maxval=max_offset_width + 1,
                                    dtype=tf.int32)'''
  offsets = torch.stack([offsets_height, offsets_width], dim=-1)

  # Loop over the batch and perform cropping.
  cropped_images = []
  for image, offset_height, offset_width in zip(batch, offsets_height,
                                                offsets_width):
    cropped_images.append(
            image[offset_height:offsets_height + target_height,
            offset_width:offset_width + target_width,
            :num_channels])
  cropped_batch = torch.stack(cropped_images)

  return cropped_batch, offsets


def random_shift(batch, max_shift_height=32, max_shift_width=32):
  """Randomly shift a batch of images (with wrap around).

  Args:
    batch: a 4-D tensor of shape [batch_size, height, width, num_channels].
    max_shift_height: an int, the maximum shift along the height dimension in
      either direction.
    max_shift_width: an int, the maximum shift along the width dimension in
      either direction

  Returns:
    a pair of 1) the shifted images in form of a tensor of shape
    [batch_size, height, width, num_channels] and 2) the random shifts of shape
    [batch_size, 2], where positive numbers mean the image was shifted
    down / right and negative numbers mean it was shifted up / left.
  """

  # Randomly sample by how much the images are being shifted.
  batch_size, _, _, _ = batch.shape
  shifts_height =((max_shift_height + 1- (-max_shift_height))*torch.rand([batch_size]) -max_shift_height).type(torch.int32)
  '''tf.random.uniform([batch_size],
                                    minval=-max_shift_height,
                                    maxval=max_shift_height + 1,
                                    dtype=tf.int32)'''
  shifts_width = ((max_shift_width + 1- (-max_shift_width))*torch.rand([batch_size]) -max_shift_width).type(torch.int32)
  '''tf.random.uniform([batch_size],
                                   minval=-max_shift_width,
                                   maxval=max_shift_width + 1,
                                   dtype=tf.int32)'''
  shifts = torch.stack([shifts_height, shifts_width], dim=-1)

  # Loop over the batch and shift the images
  shifted_images = []
  for image, shift_height, shift_width in zip(batch, shifts_height,
                                              shifts_width):
    shifted_images.append(
        torch.roll(image, shifts=[shift_height, shift_width], dims=[0, 1]))
  shifted_images = torch.stack(shifted_images)

  return shifted_images, shifts


def randomly_shift_features(feature_pyramid,
                            max_shift_height=64,
                            max_shift_width=64):
  """Randomly shift a batch of images (with wrap around).

  Args:
    feature_pyramid: a list of 4-D tensors of shape [batch_size, height, width,
      num_channels], where the first entry is at level 1 (image size / 2).
    max_shift_height: an int, the maximum shift along the height dimension in
      either direction.
    max_shift_width: an int, the maximum shift along the width dimension in
      either direction

  Returns:
    a pair of 1) a list of shifted feature images as tensors of shape
    [batch_size, height, width, num_channels] and 2) the random shifts of shape
    [batch_size, 2], where positive numbers mean the image was shifted
    down / right and negative numbers mean it was shifted up / left.
  """
  batch_size,_, height, width = feature_pyramid[0].shape
  # Image size is double the size of the features at level1 (index 0).
  height *= 2
  width *= 2

  # Transform the shift range to the size of the top level of the pyramid.
  top_level_scale = 2**len(feature_pyramid)
  max_shift_height_top_level = max_shift_height // top_level_scale
  max_shift_width_top_level = max_shift_width // top_level_scale

  # Randomly sample by how much the images are being shifted at the top level
  # and scale the shift back to level 0 (original image resolution).
  shifts_height = top_level_scale * ((max_shift_height_top_level + 1-(-max_shift_height_top_level))*torch.rand([batch_size])-max_shift_height_top_level).type(torch.int32)
  '''tf.random.uniform(
      [batch_size],
      minval=-max_shift_height_top_level,
      maxval=max_shift_height_top_level + 1,
      dtype=tf.int32)'''
  shifts_width = top_level_scale * ((max_shift_width_top_level + 1-(-max_shift_width_top_level))*torch.rand([batch_size])-max_shift_width_top_level).type(torch.int32)
  '''tf.random.uniform(
      [batch_size],
      minval=-max_shift_width_top_level,
      maxval=max_shift_width_top_level + 1,
      dtype=tf.int32)'''
  shifts = torch.stack([shifts_height, shifts_width], dim=-1)

  # Iterate over pyramid levels.
  shifted_features = []
  for level, feature_image_batch in enumerate(feature_pyramid, start=1):
    shifts_at_this_level = shifts // 2**level
    # pylint:disable=g-complex-comprehension
    shifted_features.append(
        torch.stack([
            torch.roll(
                feature_image_batch[i],
                shifts=shifts_at_this_level[i],
                dims=[0, 1]) for i in range(batch_size)
        ],
                 dim=0))

  return shifted_features, shifts.type(torch.float32)#tf.cast(shifts, dtype=tf.float32)


def zero_mask_border(mask_bhw3, patch_size):
  """Used to ignore border effects from census_transform."""
  mask_padding = patch_size // 2
  mask = mask_bhw3[:, :, mask_padding:-mask_padding, mask_padding:-mask_padding]
  return F.pad(
      input=mask,
      pad=(mask_padding,mask_padding,mask_padding,mask_padding,0,0,0,0))


def census_transform(image, patch_size):
  """The census transform as described by DDFlow.

  See the paper at https://arxiv.org/abs/1902.09145

  Args:
    image: tensor of shape (b, h, w, c)
    patch_size: int
  Returns:
    image with census transform applied
  """
  device = image.device
  intensities = TTF.rgb_to_grayscale(image,num_output_channels=1) * 255#tf.image.rgb_to_grayscale(image) * 255
  kernel = torch.reshape(
      torch.eye(patch_size * patch_size,device=device),
      (patch_size * patch_size,1,patch_size, patch_size))

  neighbors = torch.nn.functional.conv2d(input=intensities,weight=kernel,stride=(1,1),padding=(patch_size//2,patch_size//2))

  #neighbors = tf.nn.conv2d(
      #input=intensities, filters=kernel, strides=[1, 1, 1, 1], padding='SAME')
  diff = neighbors - intensities
  # Coefficients adopted from DDFlow.
  diff_norm = diff / torch.sqrt(.81 + torch.square(diff))
  return diff_norm


def soft_hamming(a_bhwk, b_bhwk, thresh=.1):
  """A soft hamming distance between tensor a_bhwk and tensor b_bhwk.

  Args:
    a_bhwk: tf.Tensor of shape (batch, height, width, features)
    b_bhwk: tf.Tensor of shape (batch, height, width, features)
    thresh: float threshold

  Returns:
    a tensor with approx. 1 in (h, w) locations that are significantly
    more different than thresh and approx. 0 if significantly less
    different than thresh.
  """
  sq_dist_bhwk = torch.square(a_bhwk - b_bhwk)
  soft_thresh_dist_bhwk = sq_dist_bhwk / (thresh + sq_dist_bhwk)
  return torch.sum(
      input=soft_thresh_dist_bhwk, dim=1, keepdim=True)




def time_it(f, num_reps=1, execute_once_before=False):
  """Times a tensorflow function in eager mode.

  Args:
    f: function with no arguments that should be timed.
    num_reps: int, number of repetitions for timing.
    execute_once_before: boolean, whether to execute the function once before
      timing in order to not count the tf.function compile time.

  Returns:
    tuple of the average time in ms and the functions output.
  """
  assert num_reps >= 1
  # Execute f once before timing it to allow tf.function to compile the graph.
  if execute_once_before:
    x = f()
  # Make sure that there is nothing still running on the GPU by waiting for the
  # completion of a bogus command.
  _ = torch.square(torch.rand([1])).numpy()
  # Time f for a number of repetitions.
  start_in_s = time.time()
  for _ in range(num_reps):
    x = f()
    # Make sure that f has finished and was not just enqueued by using another
    # bogus command. This will overestimate the computing time of f by waiting
    # until the result has been copied to main memory. Calling reduce_sum
    # reduces that overestimation.
    if isinstance(x, tuple) or isinstance(x, list):
      _ = [torch.sum(input=xi).numpy() for xi in x]
    else:
      _ = torch.sum(input=x).numpy()
  end_in_s = time.time()
  # Compute the average time in ms.
  avg_time = (end_in_s - start_in_s) * 1000. / float(num_reps)
  return avg_time, x



