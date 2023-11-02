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

"""UFlow: Unsupervised Optical Flow.

This library provides a simple interface for training and inference.
"""

import math
import sys
import time

import gin
import torch
from ckpt_manager import CheckpointManager
from py_uflow import uflow_model
from py_uflow import uflow_utils
import numpy as np
from py_uflow import uflow_flags
from torch.utils.tensorboard import SummaryWriter
from absl import flags

FLAGS = flags.FLAGS

@gin.configurable
class UFlow(object):
  """Simple interface with infer and train methods."""

  def __init__(
      self,
      checkpoint_dir='',
      summary_dir='',
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
      lr_decay_type = None,
      lr_decay_steps = None,
      gpu_learning_rate = None,
      lr_decay_after_num_steps = None
  ):
    """Instantiate a UFlow model.

    Args:
      checkpoint_dir: str, location to checkpoint model
      summary_dir: str, location to write tensorboard summary
      optimizer: str, identifier of which optimizer to use
      learning_rate: float, learning rate to use for training
      only_forward: bool, if True, only infer flow in one direction
      level1_num_layers: int, pwc architecture property
      level1_num_filters: int, pwc architecture property
      level1_num_1x1: int, pwc architecture property
      dropout_rate: float, how much dropout to use with pwc net
      build_selfsup_transformations: list of functions which transform the flow
        predicted from the raw images to be in the frame of images transformed
        by geometric_augmentation_fn
      fb_sigma_teacher: float, controls how much forward-backward flow
        consistency is needed by the teacher model in order to supervise the
        student
      fb_sigma_student: float, controls how much forward-backward consistency is
        needed by the student model in order to not receive supervision from the
        teacher model
      train_with_supervision: bool, Whether to train with ground truth flow,
        currently not supported
      train_with_gt_occlusions: bool, if True, use ground truth occlusions
        instead of predicted occlusions during training. Only works with Sintel
        which has dense ground truth occlusions.
      smoothness_edge_weighting: str, controls how smoothness penalty is
        determined
      teacher_image_version: str, which image to give to teacher model
      stop_gradient_mask: bool, whether to stop the gradient of photometric loss
        through the occlusion mask.
      selfsup_mask: str, type of selfsupervision mask to use
      normalize_before_cost_volume: bool, toggles pwc architecture property
      original_layer_sizes: bool, toggles pwc architecture property
      shared_flow_decoder: bool, toogles pwc architecutre property
      channel_multiplier: int, channel factor to use in pwc
      use_cost_volume: bool, toggles pwc architecture property
      use_feature_warp: bool, toggles pwc architecture property
      num_levels: int, how many pwc pyramid layers to use
      accumulate_flow: bool, toggles pwc architecture property
      occlusion_estimation: which type of occlusion estimation to use
      occ_weights: dict of string -> float indicating how to weight occlusions
      occ_thresholds: dict of str -> float indicating thresholds to apply for
        occlusions
      occ_clip_max: dict of string -> float indicating how to clip occlusion
      smoothness_at_level: int, which level to compute smoothness on
      use_bfloat16: bool, whether to run in bfloat16 mode.

    Returns:
      Uflow object instance.
    """
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
    self.global_step = 0
    self._feature_net = uflow_model.PWCFeaturePyramid(
        level1_num_layers=level1_num_layers,
        level1_num_filters=level1_num_filters,
        level1_num_1x1=level1_num_1x1,
        original_layer_sizes=original_layer_sizes,
        num_levels=num_levels,
        channel_multiplier=channel_multiplier,
        pyramid_resolution='half',
        use_bfloat16=use_bfloat16)
    self._flow_net = uflow_model.PWCFlow(
        dropout_rate=dropout_rate,
        normalize_before_cost_volume=normalize_before_cost_volume,
        num_levels=num_levels,
        use_feature_warp=use_feature_warp,
        use_cost_volume=use_cost_volume,
        channel_multiplier=channel_multiplier,
        accumulate_flow=accumulate_flow,
        use_bfloat16=use_bfloat16,
        shared_flow_decoder=shared_flow_decoder)
    # By default, the teacher flow and featuure models are the same as
    # the student flow and feature models.
    self._teacher_flow_net = self._flow_net
    self._teacher_feature_net = self._feature_net

    self._learning_rate = learning_rate
    self._lr_decay_type = lr_decay_type
    self._lr_decay_after_num_steps = lr_decay_after_num_steps
    self._lr_decay_steps = lr_decay_steps
    self._gpu_learning_rate = gpu_learning_rate
    self._optimizer_type = optimizer
    #self._make_or_reset_optimizer()
    self._make_or_reset_optimizer()

    # Set up checkpointing.
    #self._make_or_reset_checkpoint()
    #self.update_checkpoint_dir(checkpoint_dir)

    # Set up tensorboard log files.
    self.summary_dir = summary_dir
    if self.summary_dir:
      self.writer = SummaryWriter(log_dir=summary_dir)#tf.compat.v1.summary.create_file_writer(summary_dir)
      #self.writer.

    self._occlusion_estimation = occlusion_estimation

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

  def set_teacher_models(self, teacher_feature_net, teacher_flow_net):
    self._teacher_feature_net = teacher_feature_net
    self._teacher_flow_net = teacher_flow_net

  @property
  def featue_net(self):
    return self._feature_net

  @property
  def flow_net(self):
    return self._flow_net

  def update_checkpoint_dir(self, checkpoint_dir):
    """Changes the checkpoint directory for saving and restoring."""
    self._manager = CheckpointManager(
    assets=self._checkpoint,
    directory=checkpoint_dir,
    file_name='ckpt',
    maximum=1,
    file_format='pt'
    )

  def restore(self, reset_optimizer=False, reset_global_step=False):
    """Restores a saved model from a checkpoint."""
    status = self._manager.load()
    self._feature_net.load_state_dict(status['feature_model'])
    self._flow_net.load_state_dict(status['flow_model'])
    self._optimizer.load_state_dict(status['optimizer'])

    if reset_optimizer:
      self._make_or_reset_optimizer()
      self._make_or_reset_checkpoint()
    if reset_global_step:
      self.global_step = 0
  def get_global_step(self):
      return  self.global_step

  def save(self):
    """Saves a model checkpoint."""
    self._manager.save()

  def _make_or_reset_optimizer(self):
    if self._optimizer_type == 'adam':
      self._optimizer = torch.optim.Adam(
          params=[
                {'params': self._feature_net.parameters()},
                {'params': self._flow_net.parameters()}
            ],

          lr=self._learning_rate)
    elif self._optimizer_type == 'sgd':
      self._optimizer = torch.optim.SGD(
          params=[
                {'params': self._feature_net.parameters()},
                {'params': self._flow_net.parameters()}
            ],
          lr=self._learning_rate)

    else:
      raise ValueError('Optimizer "{}" not yet implemented.'.format(
          self._optimizer_type))

  @property
  def optimizer(self):
    return self._optimizer
  def _get_lr_scale(self):

      effective_step = torch.maximum(self.global_step - self._lr_decay_after_num_steps + 1, torch.tensor(0))
      lr_step_ratio = effective_step.type(torch.float32) / float(
          self._lr_decay_steps)
      if self._lr_decay_type == 'none' or self._lr_decay_steps <= 0:
          return self._gpu_learning_rate
      elif self._lr_decay_type == 'cosine':
          x = torch.pi * torch.minimum(lr_step_ratio, torch.tensor(1.0))
          return self._gpu_learning_rate * (torch.cos(x) + 1.0) / 2.0
      elif self._lr_decay_type == 'linear':
          return self._gpu_learning_rate * torch.maximum(1.0 - lr_step_ratio, torch.tensor(0.0))
      elif self._lr_decay_type == 'exponential':
          return self._gpu_learning_rate * 0.5 ** lr_step_ratio
      else:
          raise ValueError('Unknown lr_decay_type', self._lr_decay_type)

  def _update_learning_rate(self):
      ''' Learning rate scheduling per step '''

      lr = self._get_lr_scale()

      self.global_step += 1
      for param_group in self._optimizer.param_groups:
          param_group['lr'] = lr

  def step_and_update_lr(self):
      "Step with the inner optimizer"
      self._update_learning_rate()
      self._optimizer.step()


  def _make_or_reset_checkpoint(self):
    self._checkpoint = {
        'step': self.get_global_step(),
        'feature_model' : self._feature_net.state_dict(),
        'flow_model' : self._flow_net.state_dict(),
        'optimizer' : self._optimizer.state_dict()
    }
  # Use of tf.function breaks exporting the model, see b/138864493

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

    batch_size, seq_len, image_channels,orig_height, orig_width = images.shape

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
      images = uflow_utils.resize(
          images, input_height, input_width, is_flow=False)

    # Flatten images by folding sequence length into the batch dimension, apply
    # the feature network and undo the flattening.
    images_flattened = torch.reshape(
        input=images,
        shape = (batch_size * seq_len, image_channels, input_height, input_width))
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
      occlusion_mask = uflow_utils.resize(
          occlusion_mask, orig_height, orig_width, is_flow=False)

    # Resize and rescale flow to original resolution. This always needs to be
    # done because flow is generated at a lower resolution.
    if resize_flow_to_img_res:
      flow = uflow_utils.resize(flow, orig_height, orig_width, is_flow=True)

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
    _, _, _, occlusion_masks, _, _ = uflow_utils.compute_warps_and_occlusion(
        flows,
        self._occlusion_estimation,
        self._occ_weights,
        self._occ_thresholds,
        self._occ_clip_max,
        occlusions_are_zeros=False)
    occlusion_mask_forward = occlusion_masks[(0, 1, 'inference')][0]
    return occlusion_mask_forward

  def features_no_tf_function(self, image1, image2):
    """Runs the feature extractor portion of the model on image1 and image2."""
    images = torch.stack([image1, image2])
    # noinspection PyCallingNonCallable
    return self._feature_net(images, split_features_by_sample=True)

  def features(self, image1, image2):
    """Runs the feature extractor portion of the model on image1 and image2."""
    return self.features_no_tf_function(image1, image2)

  def train_step_no_tf_function(self,
                                batch,
                                weights=None,
                                plot_dir=None,
                                distance_metrics=None,
                                ground_truth_flow=None,
                                ground_truth_valid=None,
                                ground_truth_occlusions=None,
                                images_without_photo_aug=None,
                                occ_active=None):
    """Perform single gradient step."""
    if weights is None:
      weights = {
          'smooth2': 2.0,
          'edge_constant': 100.0,
          'census': 1.0,
      }
    else:
      # Support values and callables (e.g. to compute weights from global step).
      weights = {k: v() if callable(v) else v for k, v in weights.items()}

    losses = self._loss_and_grad(
        batch,
        weights,
        plot_dir,
        distance_metrics=distance_metrics,
        ground_truth_flow=ground_truth_flow,
        ground_truth_valid=ground_truth_valid,
        ground_truth_occlusions=ground_truth_occlusions,
        images_without_photo_aug=images_without_photo_aug,
        occ_active=occ_active)

    return losses

  def train_step(self,
                 batch,
                 weights=None,
                 distance_metrics=None,
                 ground_truth_flow=None,
                 ground_truth_valid=None,
                 ground_truth_occlusions=None,
                 images_without_photo_aug=None,
                 occ_active=None):
    """Performs a train step on the batch."""
    return self.train_step_no_tf_function(
        batch,
        weights,
        distance_metrics=distance_metrics,
        ground_truth_flow=ground_truth_flow,
        ground_truth_valid=ground_truth_valid,
        ground_truth_occlusions=ground_truth_occlusions,
        images_without_photo_aug=images_without_photo_aug,
        occ_active=occ_active)

  def train(self,
            data_it,
            num_steps,
            weights=None,
            progress_bar=True,
            plot_dir=None,
            distance_metrics=None,
            occ_active=None):
    """Trains flow from a data iterator for a number of gradient steps.

    Args:
      data_it: tf.data.Iterator that produces tensors of shape [b,3,h,w,3].
      num_steps: int, number of gradient steps to train for.
      weights: dictionary with weight for each loss.
      progress_bar: boolean flag for continuous printing of a progress bar.
      plot_dir: location to plot results or None
      distance_metrics: dictionary of which type of distance metric to use for
        photometric losses
      occ_active: dictionary of which occlusion types are active

    Returns:
      a dict that contains all losses.
    """

    # Log dictionary for storing losses of this epoch.
    log = dict()
    # Support constant lr values and callables (for learning rate schedules).
    if callable(self._learning_rate):
      log['learning-rate'] = self._learning_rate()
    else:
      log['learning-rate'] = self._learning_rate

    #print(self._learning_rate)
    start_time_data = time.time()
    #print(start_time_data)
    for _, batch in zip(range(num_steps), data_it):
      stop_time_data = time.time()

      if progress_bar:
        sys.stdout.write('..')
        sys.stdout.flush()

      # Split batch into images, occlusion masks, and ground truth flow.
      images, labels = batch[_], {}
      ground_truth_flow = labels.get('flow_uv', None)
      ground_truth_valid = labels.get('flow_valid', None)
      ground_truth_occlusions = labels.get('occlusions', None)
      images_without_photo_aug = labels.get('images_without_photo_aug', None)

      start_time_train_step = time.time()
      # Use tf.function unless intermediate results have to be plotted.
      if plot_dir is None:
        # Perform a gradient step (optimized by tf.function).
        losses = self.train_step(
            images,
            weights,
            distance_metrics=distance_metrics,
            ground_truth_flow=ground_truth_flow,
            ground_truth_valid=ground_truth_valid,
            ground_truth_occlusions=ground_truth_occlusions,
            images_without_photo_aug=images_without_photo_aug,
            occ_active=occ_active)
      else:
        # Perform a gradient step without tf.function to allow plotting.
        losses = self.train_step_no_tf_function(
            images,
            weights,
            plot_dir,
            distance_metrics=distance_metrics,
            ground_truth_flow=ground_truth_flow,
            ground_truth_valid=ground_truth_valid,
            ground_truth_occlusions=ground_truth_occlusions,
            images_without_photo_aug=images_without_photo_aug,
            occ_active=occ_active)
      stop_time_train_step = time.time()

      log_update = losses
      # Compute time in ms.
      log_update['data-time'] = (stop_time_data - start_time_data) * 1000
      log_update['train-time'] = (stop_time_train_step -
                                  start_time_train_step) * 1000

      if self.summary_dir:
          writer = SummaryWriter(self.summary_dir)
      else:
          writer = SummaryWriter()
      # Log losses and times.
      for key in log_update:
        if key in log:
          log[key].append(log_update[key])
        else:
          log[key] = [log_update[key]]
        if self.summary_dir:
          writer.add_scalar(key,log[key])#tf.summary.scalar(key, log[key])

      # Set start time for data gathering to measure data pipeline efficiency.
      start_time_data = time.time()

    for key in log:
      log[key] = torch.mean(torch.tensor(log[key]))

    if progress_bar:
      sys.stdout.write('\n')
      sys.stdout.flush()

    return log

  def _loss_and_grad(self,
                     batch,
                     weights,
                     plot_dir=None,
                     distance_metrics=None,
                     ground_truth_flow=None,
                     ground_truth_valid=None,
                     ground_truth_occlusions=None,
                     images_without_photo_aug=None,
                     occ_active=None):
    """Apply the model on the data in batch and compute the loss.

    Args:
      batch: tf.tensor of shape [b, seq, h, w, c] that holds a batch of image
        sequences.
      weights: dictionary with float entries per loss.
      plot_dir: str, directory to plot images
      distance_metrics: dict, which distance metrics to use,
      ground_truth_flow: Tensor, optional ground truth flow for first image
      ground_truth_valid: Tensor, indicates locations where gt flow is valid
      ground_truth_occlusions: Tensor, optional ground truth occlusions for
        computing loss. If None, predicted occlusions will be used.
      images_without_photo_aug: optional images without any photometric
        augmentation applied. Will be used for computing photometric losses if
        provided.
      occ_active: optional dict indicating which occlusion methods are active

    Returns:
      A tuple consisting of a tf.scalar that represents the total loss for the
      current batch, a list of gradients, and a list of the respective
      variables.
    """
    self._optimizer.zero_grad()


    losses = self.compute_loss(
          batch,
          weights,
          plot_dir,
          distance_metrics=distance_metrics,
          ground_truth_flow=ground_truth_flow,
          ground_truth_valid=ground_truth_valid,
          ground_truth_occlusions=ground_truth_occlusions,
          images_without_photo_aug=images_without_photo_aug,
          occ_active=occ_active)

    self.step_and_update_lr()
    print('loss',losses['total-loss'])
    losses['total-loss'].backward()
    return losses

  def compute_loss(self,
                   batch,
                   weights,
                   plot_dir=None,
                   distance_metrics=None,
                   ground_truth_flow=None,
                   ground_truth_valid=None,
                   ground_truth_occlusions=None,
                   images_without_photo_aug=None,
                   occ_active=None):
    """Applies the model and computes losses for a batch of image sequences."""
    # Compute only a supervised loss.
    if self._train_with_supervision:
      if ground_truth_flow is None:
        raise ValueError('Need ground truth flow to compute supervised loss.')
      flows = uflow_utils.compute_flow_for_supervised_loss(
          self._feature_net, self._flow_net, batch=batch, training=True)
      losses = uflow_utils.supervised_loss(weights, ground_truth_flow,
                                           ground_truth_valid, flows)

      losses = {key + '-loss': losses[key] for key in losses}
      return losses

    # Use possibly augmented images if non augmented version is not provided.
    if images_without_photo_aug is None:
      images_without_photo_aug = batch

    flows, selfsup_transform_fns = uflow_utils.compute_features_and_flow(
        self._feature_net,
        self._flow_net,
        batch=batch,
        batch_without_aug=images_without_photo_aug,
        training=True,
        build_selfsup_transformations=self._build_selfsup_transformations,
        teacher_feature_model=self._teacher_feature_net,
        teacher_flow_model=self._teacher_flow_net,
        teacher_image_version=self._teacher_image_version,
    )
    # Prepare images for unsupervised loss (prefer unaugmented images).
    images = dict()
    seq_len = int(batch.shape[1])

    images = {i: images_without_photo_aug[:, i] for i in range(seq_len)}


    # Warp stuff and compute occlusion.
    warps, valid_warp_masks, _, not_occluded_masks, fb_sq_diff, fb_sum_sq = uflow_utils.compute_warps_and_occlusion(
        flows,
        occlusion_estimation=self._occlusion_estimation,
        occ_weights=self._occ_weights,
        occ_thresholds=self._occ_thresholds,
        occ_clip_max=self._occ_clip_max,
        occlusions_are_zeros=True,
        occ_active=occ_active)


    # Warp images and features.
    warped_images = uflow_utils.apply_warps_stop_grad(images, warps, level=0)

    # Compute losses.
    losses = uflow_utils.compute_loss(
        weights=weights,
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
        plot_dir=plot_dir,
        distance_metrics=distance_metrics,
        smoothness_edge_weighting=self._smoothness_edge_weighting,
        stop_gradient_mask=self._stop_gradient_mask,
        selfsup_mask=self._selfsup_mask,
        ground_truth_occlusions=ground_truth_occlusions,
        smoothness_at_level=self._smoothness_at_level)
    losses = {key + '-loss': losses[key] for key in losses}
    return losses
