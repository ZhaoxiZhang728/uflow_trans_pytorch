import lightning as pl
from .endo_dataset import EndoDataset
from torch.utils.data import DataLoader, random_split
import os
import glob

class EndoTrainDataModule(pl.LightningDataModule):
	def __init__(self, data_dir, val_data=None, batch_size=1, num_workers=4, train_transform=None, val_transform=None):
		super().__init__()
		self.data_dir = data_dir
		self.val_data_dir = val_data
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.train_transform = train_transform
		self.val_transform = val_transform

	def prepare_data(self):
		pass

	def setup(self, stage):
		if stage in (None, 'fit'):
			all_data = glob.glob(os.path.join(self.data_dir,'*'))
			all_data = sorted([x for x in all_data if os.path.isdir(x)])
			if self.val_data_dir is None:
				train_partition = int(0.9 * len(all_data))
				self.train_split, self.val_split = random_split(all_data, [train_partition, len(all_data) - train_partition])
			else:
				self.train_split = all_data
				val_data = glob.glob(os.path.join(self.val_data_dir,'*'))
				self.val_split = sorted([x for x in val_data if os.path.isdir(x)])


	def train_dataloader(self):
		assert self.train_split is not None
		# train_split, train_pyr = EndoDataset(self.train_split, transform=self.train_transform)
		# return DataLoader(train_split, train_pyr, batch_size=self.batch_size, num_workers=self.num_workers,persistent_workers=True)

		train_split = EndoDataset(self.train_split, transform=self.train_transform)
		return DataLoader(train_split, batch_size=self.batch_size, num_workers=self.num_workers,persistent_workers=True)

	def val_dataloader(self):
		assert self.val_split is not None
		# val_split, val_pyr = EndoDataset(self.val_split, transform=self.val_transform)
		# return DataLoader(val_split, train_pyr, batch_size=self.batch_size, num_workers=self.num_workers,persistent_workers=True)

		val_split = EndoDataset(self.val_split, transform=self.val_transform)
		return DataLoader(val_split, batch_size=self.batch_size, num_workers=self.num_workers,persistent_workers=True)
