import lightning as pl
from .fc_dataset import FCDataset
from torch.utils.data import DataLoader, random_split
import os
import glob

class FCTrainDataModule(pl.LightningDataModule):
	def __init__(self, data_dir, batch_size=1, num_workers=4, train_transform=None, val_transform=None,shuffle =False):
		super().__init__()
		self.data_dir = data_dir
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.train_transform = train_transform
		self.val_transform = val_transform
		self.shuffle = shuffle
	def prepare_data(self):
		pass

	def train_dataloader(self):
		assert self.data_dir is not None
		# train_split, train_pyr = EndoDataset(self.train_split, transform=self.train_transform)
		# return DataLoader(train_split, train_pyr, batch_size=self.batch_size, num_workers=self.num_workers,persistent_workers=True)

		train_split = FCDataset(self.data_dir, img_transform=self.train_transform[0],flow_transform=self.train_transform[1],mode = 'train')
		return DataLoader(train_split, batch_size=self.batch_size, num_workers=self.num_workers,persistent_workers=True,shuffle=self.shuffle)

	def val_dataloader(self):
		assert self.data_dir is not None
		# val_split, val_pyr = EndoDataset(self.val_split, transform=self.val_transform)
		# return DataLoader(val_split, train_pyr, batch_size=self.batch_size, num_workers=self.num_workers,persistent_workers=True)

		val_split = FCDataset(self.data_dir, img_transform=self.train_transform[0],flow_transform=self.train_transform[1],mode = 'val')
		return DataLoader(val_split, batch_size=self.batch_size, num_workers=self.num_workers,persistent_workers=True,shuffle=self.shuffle)