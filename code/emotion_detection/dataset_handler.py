import csv
import pathlib
import torch

from itertools import islice
from collections import Counter
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg, check_integrity


class FERDataset(VisionDataset):
	'''
	Class to handle data for the FER-2013 dataset. Extends VisionDataset to be
	used by a DataLoader.

	Args:
		root (string): Root directory of dataset where directory "root/fer2013" is
		split (string, optional): The dataset split. Supports "train" (default) and "test".

		transform (callable, optional): A function/transform that takes a PIL image as input
										and returns a transformed version of the image.

		target_transform (callable, optional): A function/transform that takes in the target
											   and transforms it.
	'''
	# Class/label IDs (keys) and names (values)
	CLASSES = {
		0: 'Angry',
		1: 'Disgust',
		2: 'Fear',
		3: 'Happy',
		4: 'Sad',
		5: 'Surprise',
		6: 'Neutral',
	}

	# Dataset split types (keys) and their associated file names (values)
	_RESOURCES = {
		'train': 'train.csv',
		'val': 'train.csv',
		'test': 'test.csv',
	}

	def __init__(self, root, split='train', transform=None, target_transform=None, sample_size=None):
		# Verify the split is valid and initialize parent (VisionDataset) class
		self._split = verify_str_arg(split, 'split', self._RESOURCES.keys())
		self._sample_size = sample_size
		super().__init__(root, transform=transform, target_transform=target_transform)
		
		# Set the base folder, file name, and full file path
		self._base_folder = pathlib.Path(self.root) / 'fer2013'
		self._file_name = self._RESOURCES[self._split]
		self._data_file = self._base_folder / self._file_name

		# Verify our full file path and sample from the file if it is valid
		self.verify_data_file()
		self.sample_data_file()


	def __len__(self):
		''' Returns the number of samples '''
		return len(self._samples)

		
	def __getitem__(self, idx):
		''' Samples an item based on an index and returns its image and target '''
		image_tensor, target = self._samples[idx]
		image = Image.fromarray(image_tensor.numpy()).convert('RGB')

		if self.transform is not None:
			image = self.transform(image)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return image, target


	def verify_data_file(self):
		''' Verifies that our data file (full path to file) is valid '''
		if not check_integrity(self._data_file):
			raise RuntimeError(
					f'{self._file_name} not found in {self._base_folder}. '
					f'You can download it from: '
					f'https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data'
				)


	def sample_data_file(self):
		''' Samples all elements from our data file '''
		if self._sample_size is None:
			self._sample_size = sum(1 for line in open(self._data_file))

		with open(self._data_file, 'r', newline='') as in_file:
			sampler = islice(csv.DictReader(in_file), self._sample_size)
			self._samples = [self.sample(row) for row in sampler]


	def sample(self, row):
		''' Samples a single row in our data file, returns the image and its label '''
		pixels = [int(idx) for idx in row['pixels'].split()]
		sample_image = torch.tensor(pixels, dtype=torch.uint8).reshape(48, 48)
		sample_label = int(row['emotion']) if 'emotion' in row else None
		return (sample_image, sample_label)


	def extra_repr(self):
		''' Returns the dataset's split type, either train (default) or test '''
		return f'split={self._split}'


	def class_distribution(self):
		''' Returns the class label distribution for the dataset '''
		sample_labels = [s[1] for s in self._samples]
		label_counts = dict(Counter(sample_labels))
		return {self.CLASSES[lbl]: cnt for lbl, cnt in label_counts.items()}
