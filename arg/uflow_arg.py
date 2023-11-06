import argparse

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

"""Flags used by uflow training and evaluation."""

def parser_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--n_gpu', type=int, default=0, help='Number GPU devices to use (default 0 = CPU only)')
    parser.add_argument('--gpu', type=int, nargs='*',
                        help='Specific GPU indices to use (ignored if n_gpu=0, default [0,...,n_gpu-1])')
    parser.add_argument('--verbose', '-v', action='store_true', help='Toggle verbose')
    parser.add_argument('--predict_data_dir', type=str, default='', help='Predict data directory')
    parser.add_argument('--predict_output_dir', type=str, default='logs/predict', help='Predict output directory')
    parser.add_argument('--save_checkpoint_dir', type=str, default='/home/zhaoxizh/zhaoxizh_project/uflow/result/checkpoint',
                        help='Training log output directory')
    parser.add_argument('--load_checkpoint', type=str, default='', help='Load checkpoint')
    parser.add_argument('--check',type=bool,default=True,help='For checking whether successfully read the arg')
    parser.add_argument('--train_data_dir', type=str, default='/home/zhaoxizh/zhaoxizh_project/dataset/sintel/training/clean', help='Train data directory')
    parser.add_argument('--test_data_dir', type=str, default='/home/zhaoxizh/zhaoxizh_project/dataset/sintel/test/clean', help='Test data directory')
    parser.add_argument('--no_tf_function',type=bool,default=False,help='If True, run without'
        ' tf functions. This incurs a performance hit, but can'
        ' make debugging easier.')
    parser.add_argument('--eval_data_dir', type=str, default='', help='Eval data directory')
    parser.add_argument('--test_dir',type=str,default='',help='Test data directory')
    parser.add_argument('--plot_dir',type=str,default='',help='Plot data directory')
    #parser.add_argument('--checkpoint_dir',type=str,default='',help='Path to directory for saving and restoring checkpoints.')

    #parser.add_argument('--init_checkpoint_dir',type=str,default='',help='Path to directory for initializing from a checkpoints.')

    parser.add_argument('--frozen_teacher',type=bool,default=False,help='Whether or not to freeze the teacher model during distillation.')
    parser.add_argument('--reset_global_step',type=bool,default=True,help='Reset global step to 0 after loading from init_checkpoint')
    # General flags.
    parser.add_argument('--reset_optimizer',type=bool ,default=True, help='Reset optimizer internals after '
        'loading from init_checkpoint')
    parser.add_argument('--evaluate_during_train', type=bool,default=False,help=
                      'Whether or not to have the GPU train job perform evaluation '
                      'between epochs.')
    parser.add_argument('--from_scratch', type=bool,default=False,
                      help='Train from scratch. Do not restore the last checkpoint.')
    parser.add_argument('--no_checkpointing',type=bool,default=False,help=
                      'Do not save model checkpoints during training.')
    parser.add_argument('--epoch_length',type=int,default=1000,
                         help='Number of gradient steps per epoch.')
    parser.add_argument('--num_train_steps', type=int,default=10,help=
                         'Number of! gradient steps to train for.')
    parser.add_argument('--selfsup_after_num_steps', type=int,default=5e5,help=
                         'Number of gradient steps before self-supervision.')
    parser.add_argument('--selfsup_ramp_up_steps', type=int,default=1e5,help=
                         'Number of gradient steps for ramping up self-sup.')
    parser.add_argument('--selfsup_step_cycle', type=int,default=1e10,help=
                        'Number steps until the step counter for self-supervsion is reset.')
    parser.add_argument('--shuffle_buffer_size', type=int, default=1024,
                         help='Shuffle buffer size for training.')
    parser.add_argument('--height', type=int,default=640, help=
                        'Image height for training and evaluation.')
    parser.add_argument('--width', type=int,default=640, help=
                        'Image height for training and evaluation.')
    parser.add_argument('--crop_instead_of_resize', type=bool,default=False,help=
                        'Crops images for training instead of resizing the images.')
    parser.add_argument('--seq_len', type=bool,default=2, help=
                        'Sequence length for training flow.')
    parser.add_argument('--batch_size', type=int,default=1, help=
                        'Batch size for training flow on gpu.')
    parser.add_argument('--optimizer', type=str,default='adam', help=
                        'One of "adam", "sgd"')
    parser.add_argument('--gpu_learning_rate', type=float,default=1e-4, help=
                        'Learning rate for training UFlow on GPU.')
    parser.add_argument('--lr_decay_after_num_steps', type=int,default=0, help=
                        'decay rate of lr')
    parser.add_argument('--lr_decay_steps', type=int,default=10,help=
                        '')
    parser.add_argument('--lr_decay_type', type=str,default='none',help=
                        'One of ["none", "exponential", "linear", "gaussian"]')
    parser.add_argument('--stop_gradient_mask', type=bool,default=True, help=
                        'Whether or not to stop the gradient propagation through the occlusion mask.')
    parser.add_argument('--num_occlusion_iterations',type=int,default=1,help=
                         'If occlusion estimation is "iterative"')
    parser.add_argument('--only_forward',type=bool, default=False, help='')
    # Data augmentation (-> now gin configurable)
    parser.add_argument('--teacher_image_version',type=str, default='original',help=
                        'one of original, augmented')
    parser.add_argument(
        '--channel_multiplier',type=float,default= 1.,help=
        'Globally multiply the number of model convolution channels'
        'by this factor.')
    parser.add_argument('--num_levels',type=int,default=5,
                        help='The number of feature pyramid levels to use.')
    parser.add_argument('--use_cost_volume',type=bool,default= True, help=
                        'Whether or not to compute the cost volume.')
    parser.add_argument('--use_feature_warp', type=bool,default=True,help=
                        'Whether or not to warp the model features when computing flow.')
    parser.add_argument('--accumulate_flow', type=bool,default=True, help=
                        'Whether or not to predict a flow adjustment on each feature pyramid level.')
    parser.add_argument('--level1_num_layers',type=int,default=3, help=
                        'number of later in level1')
    parser.add_argument('--level1_num_filters',type=int,default=32, help=
                        'number of filter in level1')
    parser.add_argument('--level1_num_1x1', type=int,default=0, help=
                        'number of 1x1 filters in level1')
    parser.add_argument('--dropout_rate', type=float,default=0.1, help=
                        'Amount of level dropout.')
    parser.add_argument('--normalize_before_cost_volume', type=bool,default=True, help='')
    parser.add_argument('--original_layer_sizes', type=bool,default=False, help='')
    parser.add_argument('--shared_flow_decoder', type=bool,default=False,help= '')
    parser.add_argument('--resize_selfsup', type=bool,default=True, help='')
    parser.add_argument('--selfsup_crop_height', type=int,default=64,help=
                        'Number of pixels removed from the image at top and bottom for self-supervision.')
    parser.add_argument(
        '--selfsup_crop_width',type=int, default=64,help=
        'Number of pixels removed from the image left and right'
        'for self-supervision.')
    parser.add_argument(
        '--selfsup_max_shift', type=int,default=0,help=
        'Number of pixels removed from the image at top and bottom, left and right'
        'for self-supervision.')
    parser.add_argument(
        '--fb_sigma_teacher',type=float,default= 0.003,help=
        'Forward-backward consistency scaling constant used for self-supervision.')
    parser.add_argument(
        '--fb_sigma_student',type=float,default= 0.03,help=
        'Forward-backward consistency scaling constant used for self-supervision.')
    parser.add_argument('--selfsup_mask', type=str,default='gaussian',help=
                        'One of [gaussian, ddflow, advection]')
    parser.add_argument('--weight_photo', type=float,default=0.0, help=
                        'Weight for photometric loss.')
    parser.add_argument('--weight_ssim',type=float,default= 0.0, help=
                        'Weight for SSIM loss.')
    parser.add_argument('--weight_census',type=float,default= 1.0, help=
                        'Weight for census loss.')
    parser.add_argument('--weight_smooth1', type=float,default=0.0, help=
                        'Weight for smoothness loss.')
    parser.add_argument('--weight_smooth2', type=float,default=2.0, help=
                        'Weight for smoothness loss.')
    parser.add_argument('--smoothness_edge_constant',type=float,default=150.,help=
                       'Edge constant for smoothness loss.')
    parser.add_argument('--smoothness_edge_weighting', type=str,default='exponential',help=
                        'One of: gaussian, exponential')
    parser.add_argument('--smoothness_at_level', type=int,default=2,help= '')
    parser.add_argument('--weight_selfsup',type=float,default= 0.6, help=
                        'Weight for self-supervision loss.')
    parser.add_argument('--weight_transl_consist', type=float,default=0.0,help=
                       'Weight for loss enforcing uniform source usage.')

    # Occlusion estimation parameters
    parser.add_argument('--occlusion_estimation',type=str,default='wang',help=
                        'One of: none, brox, wang, uflow')
    parser.add_argument('--occ_after_num_steps_brox', type=int,default=0, help='')
    parser.add_argument('--occ_after_num_steps_wang', type=int,default=0, help='')
    parser.add_argument('--occ_after_num_steps_fb_abs', type=int,default=0, help='')
    parser.add_argument('--occ_after_num_steps_forward_collision',type=int,default= 0, help='')
    parser.add_argument('--occ_after_num_steps_backward_zero', type=int,default=0, help='')
    parser.add_argument('--occ_weights_fb_abs',type=float,default= 1000.0,help= '')
    parser.add_argument('--occ_weights_forward_collision', type=float,default=1000.0, help='')
    parser.add_argument('--occ_weights_backward_zero', type=float,default=1000.0, help='')
    parser.add_argument('--occ_thresholds_fb_abs', type=float,default=1.5, help='')
    parser.add_argument('--occ_thresholds_forward_collision',type=float,default= 0.4,help= '')
    parser.add_argument('--occ_thresholds_backward_zero',type=float,default= 0.25,help= '')
    parser.add_argument('--occ_clip_max_fb_abs',type=float,default=10.0, help='')
    parser.add_argument('--occ_clip_max_forward_collision',type=float,default= 5.0, help='')

    parser.add_argument('--distance_census', type=str,default='ddflow', help=
                        'Which type of distance metric to use when computing loss.')
    parser.add_argument('--distance_photo', type=str,default='robust_l1', help=
                        'Which type of distance metric to use when computing loss.')
    parser.add_argument('--use_supervision', type=bool,default=False, help=
                        'Whether or not to train with a supervised loss.')
    parser.add_argument('--resize_gt_flow_supervision', type=bool,default=True, help=
                        'Whether or not to resize ground truth flow for the supervised loss.')
    parser.add_argument('--use_gt_occlusions', type=bool,default=False,help=
                        'Whether or not to train with a ground trouth occlusion')
    # Gin params are used to specify which augmentations to perform.
    parser.add_argument('--config_file',type=str,default=None,help=
                        'Path to a Gin config file. Can be specified multiple times. '
                        'Order matters, later config files override former ones.')

    parser.add_argument('--gin_bindings', type=str,default=None,help=
                        'Newline separated list of Gin parameter bindings. Can be specified '
                        'multiple times. Overrides config from --config_file.')

    return parser.parse_args()