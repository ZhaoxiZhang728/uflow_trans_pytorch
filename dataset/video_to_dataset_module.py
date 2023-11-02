import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from dataset.video_to_dataset import Video_dataset
import torchvision as T
import torch
class TrainDatasetModule(pl.LightningDataModule):
    def __init__(self,data_dir,
                 val_data = None,
                 batch_size = 1,
                 num_workers = 4,
                 train_transform = None,
                 val_transform = None,
                 shuffle_buffer_size = 0):
        super().__init__()
        self.data_dir = data_dir
        self.val_data_dir = val_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.frames, _, _ = T.io.read_video(self.data_dir, output_format="TCHW", pts_unit='sec')
        self.n = len(self.frames)
        self.shuffle_buffer_size = shuffle_buffer_size
    def prepare_data(self):
        pass

    def shuffle(self, num):
        indexes = torch.randperm(num)

        self.frames[:num] = self.frames[indexes]

    def setup(self, stage):
        if self.shuffle_buffer_size:
            self.shuffle(self.n)
            self.shuffle(self.shuffle_buffer_size)


        if stage in (None, 'fit'):
            if self.val_data_dir is None:
                train_partition = int(0.9 * self.n)
                self.train_split, self.val_split = random_split(self.frames,[train_partition, self.n - train_partition])
            else:
                self.train_split = self.frames

    def train_dataloader(self):
        assert self.train_split is not None
        # train_split, train_pyr = EndoDataset(self.train_split, transform=self.train_transform)
        # return DataLoader(train_split, train_pyr, batch_size=self.batch_size, num_workers=self.num_workers,persistent_workers=True)

        train_split = Video_dataset(self.train_split, frame_transform=self.train_transform)
        return DataLoader(train_split, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=True,prefetch_factor=10)

    def val_dataloader(self):
        assert self.val_split is not None
        # val_split, val_pyr = EndoDataset(self.val_split, transform=self.val_transform)
        # return DataLoader(val_split, train_pyr, batch_size=self.batch_size, num_workers=self.num_workers,persistent_workers=True)

        val_split = Video_dataset(self.val_split, frame_transform=self.val_transform)
        return DataLoader(val_split, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True,prefetch_factor=10)









