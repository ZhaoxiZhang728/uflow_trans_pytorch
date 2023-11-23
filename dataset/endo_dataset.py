import os
from glob import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import List

def _read_flo(file_name: str) -> np.ndarray:
    """Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy
    # Everything needs to be in little Endian according to
    # https://vision.middlebury.edu/flow/code/flow-code/README.txt
    with open(file_name, "rb") as f:
        magic = np.fromfile(f, "c", count=4).tobytes()
        if magic != b"PIEH":
            raise ValueError("Magic number incorrect. Invalid .flo file")

        w = int(np.fromfile(f, "<i4", count=1))
        h = int(np.fromfile(f, "<i4", count=1))
        data = np.fromfile(f, "<f4", count=2 * w * h)
        return data.reshape(h, w, 2).transpose(2, 0, 1)

class EndoDataset(Dataset):
    def __init__(self, root_dir, mode=None, img_transform=None, flow_transform=None):

        self.dir = [root_dir+mode+'/clean',
                    root_dir+mode+'/albedo',
                    root_dir+mode+'/flow',
                    root_dir+mode+'/flow_viz',
                    root_dir+mode+'/occlusions']
        self.img_transform = img_transform
        self.flow_transform = flow_transform
        self._flow_list: List[str] = []
        self._abd_image_list: List[List[str]] = []
        self._flow_viz_list: List[str] = []
        self._occlusions_list: List[List[str]] = []
        self._image_list: List[List[str]] = []
        self.mode = mode
        for scene in os.listdir(self.dir[0]):
            image_list = sorted(glob(str(self.dir[0] + '/' + scene + "/*.png")))
            abd_image_list = sorted(glob(str(self.dir[1] + '/' + scene + "/*.png")))
            for i in range(len(image_list) - 1):
                self._image_list += [[image_list[i], image_list[i + 1]]]
                if self.mode == "training":
                    self._abd_image_list += [[abd_image_list[i], abd_image_list[i + 1]]]

            if self.mode == "training":
                self._flow_list += sorted(glob(str(self.dir[2] + '/' + scene + "/*.flo")))
                self._flow_viz_list += sorted(glob(str(self.dir[3] + '/' + scene + "/*.png")))
                self._occlusions_list += sorted(glob(str(self.dir[4] + '/' + scene + "/*.png")))

        self.n_samples = len(self._image_list)

    def __getitem__(self, index):
        im1 = Image.open(self._image_list[index][0])
        im2 = Image.open(self._image_list[index][1])
        # for training, randomly sample pairs from sequence
        if self.mode == "training":
            abd_im1 = Image.open(self._abd_image_list[index][0])
            abd_im2 = Image.open(self._abd_image_list[index][1])
            flo = torch.tensor(_read_flo(self._flow_list[index]))
            flo_viz = Image.open(self._flow_viz_list[index])
            occ = Image.open(self._occlusions_list[index])


            if self.img_transform is not None:
                im1 = self.img_transform(im1)
                im2 = self.img_transform(im2)
                abd_im1 = self.img_transform(abd_im1)
                abd_im2 = self.img_transform(abd_im2)
                flo_viz = self.img_transform(flo_viz)
                occ = self.img_transform(occ)
            if self.flow_transform is not None:
                flo = self.flow_transform(flo)


            dicts = {
                'images': torch.stack([abd_im1, abd_im2]),  # the output size after dataloader is [batch_size, 2, 3,height,width]
                'labels': dict(),
            }
            dicts['labels']['flow_uv'] = flo
            dicts['labels']['flow_valid'] = flo_viz
            dicts['labels']['occlusions'] = occ
            dicts['labels']['images_without_photo_aug'] = torch.stack([im1, im2])
        else:

            if self.img_transform is not None:
                im1 = self.img_transform(im1)
                im2 = self.img_transform(im2)
            dicts = {
                'images': torch.stack([im1, im2]),
                # the output size after dataloader is [batch_size, 2, 3,height,width]
                'labels': dict(),
            }
            dicts['labels']['flow_uv'] = False
            dicts['labels']['flow_valid'] = False
            dicts['labels']['occlusions'] = False
            dicts['labels']['images_without_photo_aug'] = False


        return dicts

    def __len__(self):
        return self.n_samples


if __name__ == '__main__':
    stinl_path = '/Users/zhxzhang/dataset/sintel/Sintel/'
    training_img_transforms = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor()
    ])

    training_flow_transforms = transforms.Compose([
        transforms.Resize((256, 512))
    ])
    ds = EndoDataset(root_dir=stinl_path,mode='training',img_transform=training_img_transforms,flow_transform=training_flow_transforms)

    print(ds[0]['images'].shape)
    print(ds[0]['labels']['flow_uv'].shape)
    print(ds[0]['labels']['flow_valid'].shape)
    print(ds[0]['labels']['occlusions'].shape)
    print(ds[0]['labels']['images_without_photo_aug'].shape)
