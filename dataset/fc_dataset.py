import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import FlyingChairs
import torchvision.transforms as transforms
class FCDataset(Dataset):
    def __init__(self, data_dir, mode, img_transform=None,flow_transform = None):
        self.data_dir= data_dir
        self.img_transform = img_transform
        self. flow_transform= flow_transform
        self.mode = mode
        self.ds = FlyingChairs(root=self.data_dir,split=self.mode)
        self.n_samples = len(self.ds)
        self.selection = np.array([0,1])
    def __getitem__(self, index):
        # for training, randomly sample pairs from sequence
        fn1,fn2 = np.random.choice(self.selection, size=2)
        im1 = self.ds[index][fn1]
        im2 = self.ds[index][fn2]
        flow = torch.from_numpy(self.ds[index][2])


        im1 = self.img_transform(im1)
        im2 = self.img_transform(im2)
        flow = self.flow_transform(flow)
        dicts = {
            'images': torch.stack([im1, im2])  # the output size after dataloader is [batch_size, 2, 3,height,width]
            ,
            'labels': {
                'flow_uv': flow
            }
        }
        return dicts

    def __len__(self):
        return self.n_samples



if __name__ == '__main__':
    fc_path = '/playpen/zhaoxizh/datasets/FC_dataset'
    training_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256,600))
    ])
    ds = FCDataset(fc_path ,transform=training_transforms,mode='train')
    print(len(ds))



