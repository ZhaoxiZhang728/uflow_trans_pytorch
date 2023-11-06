from dataset.video_to_dataset import Video_dataset
from absl.testing import absltest
import torch
from torch.utils.data import DataLoader
from model_operation.train_it_model import Multinet
import lightning as pl
from arg.uflow_arg import parser_arg
from functools import partial
from data_augmentation import uflow_augmentation

def get_parser():
    args = parser_arg()
    return args

class UflowTest(absltest.TestCase):
    def test_train_step(self):
        """Test a single training step."""

        args = get_parser()
        build_selfsup_transformations = partial(
            uflow_augmentation.build_selfsup_transformations,
            crop_height=args.selfsup_crop_height,
            crop_width=args.selfsup_crop_width,
            max_shift_height=args.selfsup_max_shift,
            max_shift_width=args.selfsup_max_shift,
            resize=args.resize_selfsup)
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
        #device = "cuda" if torch.cuda.is_available() else "mps"
        inputs = torch.zeros([2, 3, 256, 256])
        ds = Video_dataset(inputs)
        it = DataLoader(ds, batch_size=1)
        uflow = Multinet(
            build_selfsup_transformations=build_selfsup_transformations,
            occ_active=occ_active,
            lr_decay_steps=args.lr_decay_steps,
            lr_decay_after_num_steps=args.lr_decay_after_num_steps,
            gpu_learning_rate = args.gpu_learning_rate,
             lr_decay_type = args.lr_decay_type)


        trainer = pl.Trainer(accelerator='gpu',max_epochs=1)

        trainer.fit(uflow,it)
if __name__ == '__main__':

    absltest.main()
