
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
from model_operation.train_it_model import Multinet
import os
import lightning as pl
from dataset.endo_train_data_module import EndoTrainDataModule
from dataset.fc_data_module import FCTrainDataModule
from utils.transforms import training_img_transforms, standard_transforms,training_flow_transforms

def get_parser():
    parser = parser_arg()
    #parser.add_argument('--pre_train_data_dir', type=str, default='/playpen/zhaoxizh/datasets/FC_dataset',
    #                    help='Pre_Train data directory')
    args = parser.parse_args()
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
  if args.use_supervision:
      # Since this is the only loss in this setting, and the Adam optimizer
      # is scale invariant, the actual weight here does not matter for now.
      weights = {'supervision': 1.}
  else:
      # Note that self-supervision loss is added during training.
      weights = {
          'photo': args.weight_photo,
          'ssim': args.weight_ssim,
          'census': args.weight_census,
          'smooth1': args.weight_smooth1,
          'smooth2': args.weight_smooth2,
          'edge_constant': args.smoothness_edge_constant,
      }

      # Switch off loss-terms that have weights < 1e-7.
      weights = {
          k: v for (k, v) in weights.items() if v > 1e-7 or k == 'edge_constant'
      }
  distance_metrics = {
      'photo': args.distance_photo,
      'census': args.distance_census
  }
  def weight_selfsup_fn(current_step):
      step = current_step % args.selfsup_step_cycle
      # Start self-supervision only after a certain number of steps.
      # Linearly increase self-supervision weight for a number of steps.
      ramp_up_factor = torch.clip(
          torch.tensor(float(step - (args.selfsup_after_num_steps - 1)) /
          float(max(args.selfsup_ramp_up_steps, 1))), 0., 1.)
      return args.weight_selfsup * ramp_up_factor

  if args.plot_dir and not os.path.exists(args.plot_dir):
      print('Making new plot directory', args.plot_dir)
      os.makedirs(args.plot_dir)
  selfsup_weight_updata = lambda x:weight_selfsup_fn(x)
  uflow = Multinet(
      optimizer=args.optimizer,
      learning_rate=0.0001,
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
      plot_dir=args.plot_dir,
      weights=weights,
      distance_metrics=distance_metrics,
      occ_active=occ_active,
      lr_decay_type=args.lr_decay_type,
      lr_decay_steps= args.lr_decay_steps,
      gpu_learning_rate=args.gpu_learning_rate,
      lr_decay_after_num_steps=args.lr_decay_after_num_steps,
      selfsup_weight=selfsup_weight_updata
  )
  return uflow


def clear_gpu():
    torch.cuda.empty_cache()


def main():
    clear_gpu()
    args = get_parser()

    if args.no_tf_function:
        # tf.config.experimental_run_functions_eagerly(True)
        print('TFFUNCTION DISABLED')

    gin.parse_config_files_and_bindings(args.config_file, args.gin_bindings)
    # Make directories if they do not exist yet.

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

    uflow = create_uflow(args)

    trainset = EndoTrainDataModule(data_dir=args.train_data_dir,
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   train_transform=[training_img_transforms,training_flow_transforms],
                                   val_transform=standard_transforms)
    if args.load_checkpoint == '':
        trainer.fit(uflow, trainset)
    else:
        if args.verbose:
            print(f'Loading checkpoint {args.load_checkpoint}')
        trainer.fit(uflow, trainset, ckpt_path=args.load_checkpoint)
if __name__ == '__main__':
    main()
