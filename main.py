
from model_operation.train_it_model import Multinet
from arg.uflow_arg import parser_arg
from functools import partial
from absl import app
from absl import flags

import gin
import numpy as np
#import tensorflow as tf
import torch
from data_augmentation import uflow_augmentation
# pylint:disable=unused-import
from utils import uflow_plotting
from model_operation.train_it_model import Multinet
import os
import lightning as pl
from dataset.endo_train_data_module import EndoTrainDataModule
from dataset.endo_dataset import EndoDataset
from utils.transforms import training_transforms, standard_transforms
import glob
from torch.utils.data import DataLoader
def get_parser():
    args = parser_arg()
    return args
#def create_dataset(dir):

def create_uflow(args):
  """Build the uflow model."""

  build_selfsup_transformations = partial(
      uflow_augmentation.build_selfsup_transformations,
      crop_height=args.selfsup_crop_height,
      crop_width=args.selfsup_crop_width,
      max_shift_height=args.selfsup_max_shift,
      max_shift_width=args.selfsup_max_shift,
      resize=args.resize_selfsup)

  # Define learning rate schedules [none, cosine, linear, expoential].

  occ_weights = {
      'fb_abs': args.occ_weights_fb_abs,
      'forward_collision': args.occ_weights_forward_collision,
      'backward_zero': args.occ_weights_backward_zero,
  }
  # Switch off loss-terms that have weights < 1e-2.
  occ_weights = {k: v for (k, v) in occ_weights.items() if v > 1e-2}

  occ_thresholds = {
      'fb_abs': args.occ_thresholds_fb_abs,
      'forward_collision': args.occ_thresholds_forward_collision,
      'backward_zero': args.occ_thresholds_backward_zero,
  }
  occ_clip_max = {
      'fb_abs': args.occ_clip_max_fb_abs,
      'forward_collision': args.occ_clip_max_forward_collision,
  }
  occ_active = lambda x: {
      'uflow':
          args.occlusion_estimation == 'uflow',
      'brox':
          x > args.occ_after_num_steps_brox,
      'wang':
          x > args.occ_after_num_steps_wang,
      'wang4':
          x > args.occ_after_num_steps_wang,
      'wangthres':
          x > args.occ_after_num_steps_wang,
      'wang4thres':
          x > args.occ_after_num_steps_wang,
      'fb_abs':
          x > args.occ_after_num_steps_fb_abs,
      'forward_collision':
          x > args.occ_after_num_steps_forward_collision,
      'backward_zero':
          x > args.occ_after_num_steps_backward_zero,
  }

  uflow = Multinet(
      optimizer=args.optimizer,
      learning_rate=0.00001,
      only_forward=args.only_forward,
      level1_num_layers=args.level1_num_layers,
      level1_num_filters=args.level1_num_filters,
      level1_num_1x1=args.level1_num_1x1,
      dropout_rate=args.dropout_rate,
      build_selfsup_transformations=build_selfsup_transformations,
      fb_sigma_teacher=args.fb_sigma_teacher,
      fb_sigma_student=args.fb_sigma_student,
      train_with_supervision=args.use_supervision,
      train_with_gt_occlusions=args.use_gt_occlusions,
      smoothness_edge_weighting=args.smoothness_edge_weighting,
      teacher_image_version=args.teacher_image_version,
      stop_gradient_mask=args.stop_gradient_mask,
      selfsup_mask=args.selfsup_mask,
      normalize_before_cost_volume=args.normalize_before_cost_volume,
      original_layer_sizes=args.original_layer_sizes,
      shared_flow_decoder=args.shared_flow_decoder,
      channel_multiplier=args.channel_multiplier,
      num_levels=args.num_levels,
      use_cost_volume=args.use_cost_volume,
      use_feature_warp=args.use_feature_warp,
      accumulate_flow=args.accumulate_flow,
      occlusion_estimation=args.occlusion_estimation,
      occ_weights=occ_weights,
      occ_thresholds=occ_thresholds,
      occ_clip_max=occ_clip_max,
      smoothness_at_level=args.smoothness_at_level,
      occ_active=occ_active,
      lr_decay_type=args.lr_decay_type,
      lr_decay_steps= args.lr_decay_steps,
      gpu_learning_rate=args.gpu_learning_rate,
      lr_decay_after_num_steps=args.lr_decay_after_num_steps
  )
  return uflow
def create_frozen_teacher_models(args,uflow):
  """Create a frozen copy of the current uflow model."""
  uflow_copy = create_uflow()
  teacher_feature_model = uflow_copy.feature_model
  teacher_flow_model = uflow_copy.flow_model
  # need to create weights in teacher models by calling them
  bogus_input1 = torch.randn(args.batch_size, args.height,
                                 args.width, 3).type(torch.float32)
  bogus_input2 = torch.randn(args.batch_size, args.height,
                                 args.width, 3).type(torch.float32)
  existing_model_output = uflow.feature_model(
      bogus_input1, split_features_by_sample=False)
  _ = teacher_feature_model(bogus_input1, split_features_by_sample=False)
  teacher_feature_model.set_weights(uflow.feature_model.get_weights())
  teacher_output1 = teacher_feature_model(
      bogus_input1, split_features_by_sample=False)
  teacher_output2 = teacher_feature_model(
      bogus_input2, split_features_by_sample=False)

  # check that both feature models have the same output
  assert np.max(existing_model_output[-1].numpy() -
                teacher_output1[-1].numpy()) < .01
  existing_model_flow = uflow.flow_model(
      teacher_output1, teacher_output2, training=False)
  _ = teacher_flow_model(teacher_output1, teacher_output2, training=False)
  teacher_flow_model.set_weights(uflow.flow_model.get_weights())
  teacher_flow = teacher_flow_model(
      teacher_output1, teacher_output2, training=False)
  # check that both flow models have the same output
  assert np.max(existing_model_flow[-1].numpy() -
                teacher_flow[-1].numpy()) < .01
  # Freeze the teacher models.
  for layer in teacher_feature_model.layers:
    layer.trainable = False
  for layer in teacher_flow_model.layers:
    layer.trainable = False

  return teacher_feature_model, teacher_flow_model

def check_model_frozen(feature_model, flow_model, args,prev_flow_output=None):
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

def main():

    args = get_parser()

    if args.no_tf_function:
        # tf.config.experimental_run_functions_eagerly(True)
        print('TFFUNCTION DISABLED')

    gin.parse_config_files_and_bindings(args.config_file, args.gin_bindings)
    # Make directories if they do not exist yet.

    if args.plot_dir and not os.path.exists(args.plot_dir):
        print('Making new plot directory', args.plot_dir)
        os.makedirs(args.plot_dir)

    if not os.path.exists(args.save_checkpoint_dir):
        os.makedirs(args.save_checkpoint_dir)
    if args.n_gpu > 0:
        if args.gpu is None or len(args.gpu) != args.n_gpu:
            if args.verbose:
                print(f'Got n_gpu={args.n_gpu} but gpu={args.gpu}, setting to {list(range(args.n_gpu))}')
            args.gpu = list(range(args.n_gpu))
        trainer = pl.Trainer(accelerator='gpu', devices=args.gpu, default_root_dir=args.save_checkpoint_dir,
                             check_val_every_n_epoch=1, max_epochs=args.num_train_steps)
    else:
        trainer = pl.Trainer(accelerator='gpu',
                            default_root_dir=args.save_checkpoint_dir,
                             check_val_every_n_epoch=1,
                             max_epochs=args.num_train_steps)

    if args.train_data_dir == '':
        if args.verbose:
            print(f'No train data specified, skipping training.')
        # model init
        if args.load_checkpoint == '':
            if args.verbose:
                print(f'No checkpoint set, creating new model.')
            uflow = create_uflow(args)
        else:
            if args.verbose:
                print(f'Loading model from checkpoint {args.load_checkpoint}')
            uflow = Multinet.load_from_checkpoint(args.load_checkpoint)
    else:
        uflow = create_uflow(args)
        trainset = EndoTrainDataModule(data_dir=args.train_data_dir,
                                       val_data=args.test_data_dir,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers,
                                       train_transform=training_transforms,
                                       val_transform=standard_transforms)
        if args.load_checkpoint == '':
            trainer.fit(uflow, trainset)
        else:
            if args.verbose:
                print(f'Loading checkpoint {args.load_checkpoint}')
            trainer.fit(uflow, trainset, ckpt_path=args.load_checkpoint)

    if args.test_data_dir == '':
        if args.verbose:
            print(f'No test data specified, skipping testing.')
    else:
        all_data = glob.glob(os.path.join(args.test_data_dir, '*'))
        all_data = sorted([x for x in all_data if os.path.isdir(x)])
        test_dataset = EndoDataset(all_data, transform=standard_transforms)
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers)
        trainer.test(uflow, test_loader)
if __name__ == '__main__':
    main()
