import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class EndoDataset(Dataset):
    def __init__(self, sequences, transform=None):
        self.sequences = sequences
        self.transform = transform

        self.n_samples = 0
        self.seq_vals = np.empty(len(self.sequences), dtype=np.int32)
        for i,sequence in enumerate(self.sequences):
            n = len(glob.glob(os.path.join(sequence,'*.png'))+glob.glob(os.path.join(sequence,'*.jpg')))
            self.n_samples += int(n * (n-1) / 2)
            self.seq_vals[i] = self.n_samples

    def __getitem__(self, index):
        sequence = self.index_to_sequence(index)
        # for training, randomly sample pairs from sequence
        fn = np.random.choice(sequence,size=2)
        im1 = Image.open(fn[0])
        im2 = Image.open(fn[1])
        if self.transform:
            im1 = self.transform(im1)
            im2 = self.transform(im2)

        # num_levels = 3
        # im1_pyr = [im1]        
        # im1_pyr.append(layer1(im1))
        # im1_pyr.append(layer2(im1))

        # im2_pyr = [im2]
        # im2_pyr.append(layer1(im2))
        # im2_pyr.append(layer2(im2))
        dicts = {
            'images': torch.stack([im1, im2])  # the output size after dataloader is [batch_size, 2, 3,height,width]
            ,
            'labels': {
                'flow_uv': False,
                'flow_valid': False,
                'occlusions': False,
                'images_without_photo_aug': False
            }
        }
        return dicts

    def __len__(self):
        return self.n_samples

    def index_to_sequence(self, index):
        seq_index = np.amin(np.where(index <= self.seq_vals))
        sequence = self.sequences[seq_index]
        return glob.glob(os.path.join(sequence,'*.png'))+glob.glob(os.path.join(sequence,'*.jpg'))


if __name__ == '__main__':
    stinl_path = 'E:/dataset/sintel/training/clean'
    all_data = glob.glob(os.path.join(stinl_path, '*'))
    all_data = sorted([x for x in all_data if os.path.isdir(x)])
    training_transforms = transforms.Compose([
        transforms.Resize((256,600)),
        transforms.ToTensor()
    ])
    ds = EndoDataset(all_data,transform=training_transforms)
    print(ds[0]['images'].shape)
