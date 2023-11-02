import lightning as pl
import torch
import torch.nn as nn
from model_operation.feature_model import PWCFeaturePyramid
from model_operation.flow_model import PWCFlow
from utils.uflow_utils import compute_features_and_flow,compute_flow_for_supervised_loss,apply_warps_stop_grad,compute_warps_and_occlusion,resize
from model_operation.loss import supervised_loss,compute_loss
import math
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
#from model_operation.loss


class Multinet(pl.LightningModule):
    def __init__(
      self,
      optimizer='adam',
      learning_rate=0.0002,
      only_forward=False,
      level1_num_layers=3,
      level1_num_filters=32,
      level1_num_1x1=0,
      dropout_rate=.25,
      build_selfsup_transformations=None,
      fb_sigma_teacher=0.003,
      fb_sigma_student=0.03,
      train_with_supervision=False,
      train_with_gt_occlusions=False,
      smoothness_edge_weighting='gaussian',
      teacher_image_version='original',
      stop_gradient_mask=True,
      selfsup_mask='gaussian',
      normalize_before_cost_volume=True,
      original_layer_sizes=False,
      shared_flow_decoder=False,
      channel_multiplier=1,
      use_cost_volume=True,
      use_feature_warp=True,
      num_levels=5,
      accumulate_flow=True,
      occlusion_estimation='wang',
      occ_weights=None,
      occ_thresholds=None,
      occ_clip_max=None,
      smoothness_at_level=2,
      use_bfloat16=False,
      weights = None,
      distance_metrics = None,
      occ_active = None,
      plot_dir = None,
      lr_decay_after_num_steps = None,
      lr_decay_steps = None,
      gpu_learning_rate = None,
      lr_decay_type = None
    ):
        super().__init__()
        self._lr_decay_type = lr_decay_type
        self._lr_decay_steps = lr_decay_steps
        self._gpu_learning_rate = gpu_learning_rate
        self._lr_decay_after_num_steps = lr_decay_after_num_steps
        self._optimizer = optimizer,
        self._learning_rate = learning_rate
        self._only_forward = only_forward
        self._build_selfsup_transformations = build_selfsup_transformations
        self._fb_sigma_teacher = fb_sigma_teacher
        self._fb_sigma_student = fb_sigma_student
        self._train_with_supervision = train_with_supervision
        self._train_with_gt_occlusions = train_with_gt_occlusions
        self._smoothness_edge_weighting = smoothness_edge_weighting
        self._smoothness_at_level = smoothness_at_level
        self._teacher_flow_net = None
        self._teacher_feature_net = None
        self._teacher_image_version = teacher_image_version
        self._stop_gradient_mask = stop_gradient_mask
        self._selfsup_mask = selfsup_mask
        self._num_levels = num_levels
        self._feature_net = PWCFeaturePyramid(
            level1_num_layers=level1_num_layers,
            level1_num_filters=level1_num_filters,
            level1_num_1x1=level1_num_1x1,
            original_layer_sizes=original_layer_sizes,
            num_levels=num_levels,
            channel_multiplier=channel_multiplier,
            pyramid_resolution='half',
            use_bfloat16=use_bfloat16)

        self._flow_net = PWCFlow(
            dropout_rate=dropout_rate,
            normalize_before_cost_volume=normalize_before_cost_volume,
            num_levels=num_levels,
            use_feature_warp=use_feature_warp,
            use_cost_volume=use_cost_volume,
            channel_multiplier=channel_multiplier,
            accumulate_flow=accumulate_flow,
            use_bfloat16=use_bfloat16,
            shared_flow_decoder=shared_flow_decoder
        )

        if self._teacher_feature_net is None:
            self._teacher_feature_net = PWCFeaturePyramid(
                        level1_num_layers=level1_num_layers,
                        level1_num_filters=level1_num_filters,
                        level1_num_1x1=level1_num_1x1,
                        original_layer_sizes=original_layer_sizes,
                        num_levels=num_levels,
                        channel_multiplier=channel_multiplier,
                        pyramid_resolution='half',
                        use_bfloat16=use_bfloat16
            )
            self._teacher_feature_net.load_state_dict(self._feature_net.state_dict())
            self._teacher_feature_net.freeze()


        if self._teacher_flow_net is None:
            self._teacher_flow_net = PWCFlow(
                dropout_rate=dropout_rate,
                normalize_before_cost_volume=normalize_before_cost_volume,
                num_levels=num_levels,
                use_feature_warp=use_feature_warp,
                use_cost_volume=use_cost_volume,
                channel_multiplier=channel_multiplier,
                accumulate_flow=accumulate_flow,
                use_bfloat16=use_bfloat16,
                shared_flow_decoder=shared_flow_decoder
            )
            self._teacher_flow_net.load_state_dict(self._flow_net.state_dict())
            self._teacher_flow_net.freeze()

        self._occlusion_estimation = occlusion_estimation
        self.plot_dir = plot_dir
        if occ_weights is None:
            occ_weights = {
                'fb_abs': 1.0,
                'forward_collision': 1.0,
                'backward_zero': 10.0
            }
        self._occ_weights = occ_weights

        if occ_thresholds is None:
            occ_thresholds = {
                'fb_abs': 1.5,
                'forward_collision': 0.4,
                'backward_zero': 0.25
            }
        self._occ_thresholds = occ_thresholds

        if occ_clip_max is None:
            occ_clip_max = {'fb_abs': 10.0, 'forward_collision': 5.0}
        self._occ_clip_max = occ_clip_max
        self.weights = weights
        if self.weights is None:
            self.weights = {
            'smooth2': 2.0,
            'edge_constant': 100.0,
            'census': 1.0,
            }
        self.distance_metrics = distance_metrics
        self.occ_active  = occ_active

    def forward(self, image):
        return self._feature_net(image)

    def forward_backprop(self, batch):


        images, labels = batch['images'], batch['labels']

        ground_truth_flow = labels['flow_uv'] if labels['flow_uv'] else None
        ground_truth_valid = labels['flow_valid'] if labels['flow_valid'] else None
        ground_truth_occlusions = labels['occlusions'] if labels['occlusions'] else None
        images_without_photo_aug = labels['images_without_photo_aug'] if labels['images_without_photo_aug'] else None

        if self._train_with_supervision:
            if ground_truth_flow is None:
                raise ValueError('Need ground truth flow to compute supervised loss.')
            if self._train_with_supervision:
                if ground_truth_flow is None:
                    raise ValueError('Need ground truth flow to compute supervised loss.')
                flows = compute_flow_for_supervised_loss(
                    self._feature_net, self._flow_net, batch=images, training=True)
                losses = supervised_loss(self.weights, ground_truth_flow,
                                                     ground_truth_valid, flows)

                losses = {key + '-loss': losses[key] for key in losses}
                return losses['total-loss']
        else:
            if images_without_photo_aug is None:
                images_without_photo_aug = images
            flows, selfsup_transform_fns = compute_features_and_flow(
                self._feature_net,
                self._flow_net,
                batch=images,
                batch_without_aug=images_without_photo_aug,
                training=True,
                build_selfsup_transformations=self._build_selfsup_transformations,
                teacher_feature_model=self._teacher_feature_net,
                teacher_flow_model=self._teacher_flow_net,
                teacher_image_version=self._teacher_image_version,
            )
            # Prepare images for unsupervised loss (prefer unaugmented images).
            images = dict()
            seq_len = int(batch['images'].shape[1])

            images = {i: images_without_photo_aug[:, i] for i in range(seq_len)}
            occ_active = self.get_occ_active() if self.occ_active is not None else None
            # Warp stuff and compute occlusion.
            warps, valid_warp_masks, _, not_occluded_masks, fb_sq_diff, fb_sum_sq = compute_warps_and_occlusion(
                flows,
                occlusion_estimation=self._occlusion_estimation,
                occ_weights=self._occ_weights,
                occ_thresholds=self._occ_thresholds,
                occ_clip_max=self._occ_clip_max,
                occlusions_are_zeros=True,
                occ_active=occ_active)

            # Warp images and features.
            warped_images = apply_warps_stop_grad(images, warps, level=0)
            losses = compute_loss(weights=self.weights,
                                    images=images,
                                    flows=flows,
                                    warps=warps,
                                    valid_warp_masks=valid_warp_masks,
                                    not_occluded_masks=not_occluded_masks,
                                    fb_sq_diff=fb_sq_diff,
                                    fb_sum_sq=fb_sum_sq,
                                    warped_images=warped_images,
                                    only_forward=self._only_forward,
                                    selfsup_transform_fns=selfsup_transform_fns,
                                    fb_sigma_teacher=self._fb_sigma_teacher,
                                    fb_sigma_student=self._fb_sigma_student,
                                    plot_dir=self.plot_dir,
                                    distance_metrics=self.distance_metrics,
                                    smoothness_edge_weighting=self._smoothness_edge_weighting,
                                    stop_gradient_mask=self._stop_gradient_mask,
                                    selfsup_mask=self._selfsup_mask,
                                    ground_truth_occlusions=ground_truth_occlusions,
                                    smoothness_at_level=self._smoothness_at_level)
            losses = {key + '-loss': losses[key] for key in losses}
            return losses['total-loss']

    def training_step(self, batch, batch_idx):
        loss = self.forward_backprop(batch)
        self.log('train-loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward_backprop(batch)
        self.log('val-loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.forward_backprop(batch)
        self.log('test-loss', loss)
        return loss

    def configure_optimizers(self):
        if self._optimizer == ('adam',):
            optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)
        if self._optimizer == ('sgd',):
            optimizer = torch.optim.SGD(self.parameters(), lr=self._learning_rate)

        lr_scheduler = LambdaLR(optimizer, lr_lambda=self.lr_lambda)
        #scheduler = LambdaLR(optimizer=optimizer, lr_lambda= [])

        return [optimizer], [lr_scheduler]
    def get_occ_active(self):
        current_step = self.global_step

        # Set which occlusion estimation methods could be active at this point.
        # (They will only be used if occlusion_estimation is set accordingly.)
        occ_active = self.occ_active(current_step)
        return occ_active

    def lr_lambda(self,epoch):

        effective_step = np.maximum(epoch - self._lr_decay_after_num_steps + 1, 0)
        lr_step_ratio = effective_step.astype(np.float32) / float(
            self._lr_decay_steps)
        if self._lr_decay_type == 'none' or self._lr_decay_steps <= 0:
            return self._gpu_learning_rate
        elif self._lr_decay_type == 'cosine':
            m = np.pi * np.minimum(lr_step_ratio, 1.0)
            return self._gpu_learning_rate * (np.cos(m) + 1.0) / 2.0
        elif self._lr_decay_type == 'linear':
            return self._gpu_learning_rate * np.maximum(1.0 - lr_step_ratio, 0.0)
        elif self._lr_decay_type == 'exponential':
            return self._gpu_learning_rate * 0.5 ** lr_step_ratio
        else:
            raise ValueError('Unknown lr_decay_type', self._lr_decay_type)
            

    def batch_infer_no_tf_function(self,
                                   images,
                                   input_height=None,
                                   input_width=None,
                                   resize_flow_to_img_res=True,
                                   infer_occlusion=False):
        """Infers flow from two images.

        Args:
          images: tf.tensor of shape [batchsize, 2, height, width, 3].
          input_height: height at which the model should be applied if different
            from image height.
          input_width: width at which the model should be applied if different from
            image width
          resize_flow_to_img_res: bool, if True, return the flow resized to the same
            resolution as (image1, image2). If False, return flow at the whatever
            resolution the model natively predicts it.
          infer_occlusion: bool, if True, return both flow and a soft occlusion
            mask, else return just flow.

        Returns:
          Optical flow for each pixel in image1 pointing to image2.
        """

        batch_size, seq_len, image_channels, orig_height, orig_width = images.shape

        if input_height is None:
            input_height = orig_height
        if input_width is None:
            input_width = orig_width

        # Ensure a feasible computation resolution. If specified size is not
        # feasible with the model, change it to a slightly higher resolution.
        divisible_by_num = pow(2.0, self._num_levels)
        if (input_height % divisible_by_num != 0 or
                input_width % divisible_by_num != 0):
            print('Cannot process images at a resolution of ' + str(input_height) +
                  'x' + str(input_width) + ', since the height and/or width is not a '
                                           'multiple of ' + str(divisible_by_num) + '.')
            # compute a feasible resolution
            input_height = int(
                math.ceil(float(input_height) / divisible_by_num) * divisible_by_num)
            input_width = int(
                math.ceil(float(input_width) / divisible_by_num) * divisible_by_num)
            print('Inference will be run at a resolution of ' + str(input_height) +
                  'x' + str(input_width) + '.')

        # Resize images to desired input height and width.
        if input_height != orig_height or input_width != orig_width:
            images = resize(
                images, input_height, input_width, is_flow=False)

        # Flatten images by folding sequence length into the batch dimension, apply
        # the feature network and undo the flattening.
        images_flattened = torch.reshape(
            input=images,
            shape=(batch_size * seq_len, image_channels, input_height, input_width))
        # noinspection PyCallingNonCallable
        features_flattened = self._feature_net(
            images_flattened, split_features_by_sample=False)
        features = [
            torch.reshape(input=f, shape=tuple([batch_size, seq_len] + list(f.shape[1:])))
            for f in features_flattened
        ]

        features1, features2 = [[f[:, i] for f in features] for i in range(2)]

        # Compute flow in frame of image1.
        # noinspection PyCallingNonCallable
        flow = self._flow_net(features1, features2, training=False)[0]

        if infer_occlusion:
            # noinspection PyCallingNonCallable
            flow_backward = self._flow_net(features2, features1, training=False)[0]
            occlusion_mask = self.infer_occlusion(flow, flow_backward)
            occlusion_mask = resize(
                occlusion_mask, orig_height, orig_width, is_flow=False)

        # Resize and rescale flow to original resolution. This always needs to be
        # done because flow is generated at a lower resolution.
        if resize_flow_to_img_res:
            flow = resize(flow, orig_height, orig_width, is_flow=True)

        if infer_occlusion:
            return flow, occlusion_mask

        return flow

    def infer(self,
              image1,
              image2,
              input_height=None,
              input_width=None,
              resize_flow_to_img_res=True,
              infer_occlusion=False):
        return self.infer_no_tf_function(image1, image2, input_height, input_width,
                                         resize_flow_to_img_res, infer_occlusion)

    def batch_infer(self,
                    images,
                    input_height=None,
                    input_width=None,
                    resize_flow_to_img_res=True,
                    infer_occlusion=False):
        return self.batch_infer_no_tf_function(images, input_height, input_width,
                                               resize_flow_to_img_res,
                                               infer_occlusion)

    def infer_occlusion(self, flow_forward, flow_backward):
        """Gets a 'soft' occlusion mask from the forward and backward flow."""

        flows = {
            (0, 1, 'inference'): [flow_forward],
            (1, 0, 'inference'): [flow_backward],
        }
        _, _, _, occlusion_masks, _, _ = compute_warps_and_occlusion(
            flows,
            self._occlusion_estimation,
            self._occ_weights,
            self._occ_thresholds,
            self._occ_clip_max,
            occlusions_are_zeros=False)
        occlusion_mask_forward = occlusion_masks[(0, 1, 'inference')][0]
        return occlusion_mask_forward

    def infer_no_tf_function(self,
                             image1,
                             image2,
                             input_height=None,
                             input_width=None,
                             resize_flow_to_img_res=True,
                             infer_occlusion=False):
        """Infer flow for two images.

        Args:
          image1: tf.tensor of shape [height, width, 3].
          image2: tf.tensor of shape [height, width, 3].
          input_height: height at which the model should be applied if different
            from image height.
          input_width: width at which the model should be applied if different from
            image width
          resize_flow_to_img_res: bool, if True, return the flow resized to the same
            resolution as (image1, image2). If False, return flow at the whatever
            resolution the model natively predicts it.
          infer_occlusion: bool, if True, return both flow and a soft occlusion
            mask, else return just flow.

        Returns:
          Optical flow for each pixel in image1 pointing to image2.
        """

        results = self.batch_infer_no_tf_function(
            torch.stack([image1, image2])[None],
            input_height=input_height,
            input_width=input_width,
            resize_flow_to_img_res=resize_flow_to_img_res,
            infer_occlusion=infer_occlusion)

        # Remove batch dimension from all results.
        if isinstance(results, (tuple, list)):
            return [x[0] for x in results]
        else:
            return results[0]

    def check_model_frozen(feature_model, flow_model, args, prev_flow_output=None):
        """Check that a frozen model isn't somehow changing over time."""
        state = np.random.RandomState(40)
        input1 = state.randn(args.batch_size, args.height, args.width,
                             3).astype(np.float32)
        input2 = state.randn(args.batch_size, args.height, args.width,
                             3).astype(np.float32)
        feature_output1 = feature_model(input1, split_features_by_sample=False)
        feature_output2 = feature_model(input2, split_features_by_sample=False)
        flow_output = flow_model(feature_output1, feature_output2, training=False)
        if prev_flow_output is None:
            return flow_output
        for f1, f2 in zip(prev_flow_output, flow_output):
            assert np.max(f1.numpy() - f2.numpy()) < .01









