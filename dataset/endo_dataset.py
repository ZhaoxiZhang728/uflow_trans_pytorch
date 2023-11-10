import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms

class EndoDataset(Dataset):
    def __init__(self, sequences, transform=None):
        self.sequences = sequences
        self.transform = transform

        self.seq_lens = len(self.sequences)

        self.n_samples = 0
        for i, sequence in enumerate(self.sequences):
            n = len(glob.glob(os.path.join(sequence, '*.png')) + glob.glob(os.path.join(sequence, '*.jpg')))
            self.n_samples += int(n)

    def __getitem__(self, index):
        sequence = self.index_to_sequence(index)
        # for training, randomly sample pairs from sequence
        fn = np.random.choice(sequence,size=2)
        im1 = Image.open(fn[0])
        im2 = Image.open(fn[1])
        if self.transform:
            im1 = self.transform(im1)
            im2 = self.transform(im2)

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

    def index_to_sequence(self, index):# index doesn't have any meaning.
        seq_index = np.amin(np.random.randint(0, self.seq_lens))
        sequence = self.sequences[seq_index]
        return glob.glob(os.path.join(sequence,'*.png'))+glob.glob(os.path.join(sequence,'*.jpg'))


if __name__ == '__main__':
    stinl_path = '/Users/zhxzhang/dataset/sintel/MPI-Sintel-complete/training/clean'
    all_data = glob.glob(os.path.join(stinl_path, '*'))
    all_data = sorted([x for x in all_data if os.path.isdir(x)])
    training_transforms = transforms.Compose([
        transforms.Resize((256,600)),
        transforms.ToTensor()
    ])
    ds = EndoDataset(all_data,transform=training_transforms)


