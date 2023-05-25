import os
import argparse
import torch
import numpy as np

from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler


def parse_args():
	'''
	This function creates argument parser and parses the scrip input arguments.
	This is the most common way to define input arguments in python. It is used
	by train.py and human_eval.py

	To change the parameters, pass them to the script, for example:

	python cli/train.py \
		--output_dir output_dir \
		--weight_decay 0.01
	
	Default arguments have the meaning of being a reasonable default value, not of the last arguments used.
	'''
	parser = argparse.ArgumentParser(description='Fine-tune image classification model')

	# Required arguments
	parser.add_argument(
		'--data_dir',
		type=str,
		default='dataset',
		help='Where the dataset is stored.',
	)
	parser.add_argument(
		'--output_dir',
		type=str,
		default='outputs',
		help='Where to store the final model.',
	)
	parser.add_argument(
		'--pretrained_model_name',
		type=str,
		default='microsoft/resnet-18',
		help='Name of pretrained model to be used.',
	)
	parser.add_argument(
		'--dataset_dir',
		type=str,
		default='dataset',
		help='Directory where the dataset is stored.',
	)
	parser.add_argument(
		'--test_for_val',
		default=False,
		action='store_true',
		help='Use the test set for validation. By default, a subset of training data is used.',
	)
	parser.add_argument(
		'--test_type',
		type=str,
		default='public_test',
		help='The testing data split type to use.',
		choices=['public_test', 'private_test'],
	)
	parser.add_argument(
		'--train_val_size',
		type=int,
		default=None,
		help='Combined size (# samples) of the training and validation set.',
	)
	parser.add_argument(
		'--test_size',
		type=int,
		default=None,
		help='Size (# samples) of the test set.',
	)
	parser.add_argument(
		'--percent_train',
		type=float,
		default=0.8,
		help='Percentage of the data to use for training (train_val_size * percent_train).',
	)
	parser.add_argument(
		'--debug',
		default=False,
		action='store_true',
		help='Whether to use a small subset of the dataset for debugging.',
	)

	# Training arguments
	parser.add_argument(
		'--device',
		default='cuda' if torch.cuda.is_available() else 'cpu',
		help='Device (cuda or cpu) on which the code should run',
	)
	parser.add_argument(
		'--batch_size',
		type=int,
		default=128,
		help='Batch size (per device) for the training dataloader.',
	)
	parser.add_argument(
		'--learning_rate',
		type=float,
		default=5e-4,
		help='Initial learning rate (after the potential warmup period) to use.',
	)
	parser.add_argument(
		'--weight_decay',
		type=float,
		default=0.0,
		help='Weight decay to use.',
	)
	parser.add_argument(
		'--num_train_epochs',
		type=int,
		default=15,
		help='Total number of training epochs to perform.',
	)
	parser.add_argument(
		'--eval_every_steps',
		type=int,
		default=40,
		help='Perform evaluation every n network updates.',
	)
	parser.add_argument(
		'--logging_steps',
		type=int,
		default=20,
		help='Compute and log training batch metrics every n steps.',
	)
	parser.add_argument(
		'--checkpoint_every_steps',
		type=int,
		default=500,
		help='Save model checkpoint every n steps.',
	)
	parser.add_argument(
		'--max_train_steps',
		type=int,
		default=None,
		help='Total number of training steps to perform. If provided, overrides num_train_epochs.',
	)
	parser.add_argument(
		'--lr_scheduler_type',
		type=str,
		default='linear',
		help='The scheduler type to use.',
		choices=['no_scheduler', 'linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'],
	)
	parser.add_argument(
		'--num_warmup_steps',
		type=int,
		default=0,
		help='Number of steps for the warmup in the lr scheduler.',
	)
	parser.add_argument(
		'--seed',
		type=int,
		default=42,
		help='A seed for reproducible training.',
	)
	
	# Weights and biases (wandb) arguments
	parser.add_argument(
		'--use_wandb',
		default=False,
		action='store_true',
		help='Whether to enable usage/logging for the wandb_project.',
	)
	parser.add_argument(
		'--wandb_project', 
		default='emotion_detection',
		help='wandb project name to log metrics to'
	)

	# Live demo arguments
	parser.add_argument(
		'--demo_cascade_file',
		type=str,
		default=os.path.join('.', 'haarcascade_frontalface_default.xml'),
		help='The cascade file to use for facial detection in the demo.'
	)
	parser.add_argument(
		'--demo_window_width',
		type=int,
		default=720,
		help='The width (in pixels) of the live demo window.',
	)
	parser.add_argument(
		'--demo_window_height',
		type=int,
		default=480,
		help='The height (in pixels) of the live demo window.',
	)
	parser.add_argument(
		'--demo_video_width',
		type=int,
		default=425,
		help='The width (in pixels) of the live demo video.',
	)
	parser.add_argument(
		'--demo_video_height',
		type=int,
		default=425,
		help='The height (in pixels) of the live demo video.',
	)
	parser.add_argument(
		'--demo_font_type',
		type=str,
		default='monospace',
		help='The type of font to use throughout the demo.',
	)
	parser.add_argument(
		'--demo_font_size',
		type=int,
		default=15,
		help='The font size to use throughout the demo.',
	)

	args = parser.parse_args()
	return args


def get_norm(feature_extractor):
	''' Returns the normalization transform '''
	norm = transforms.Normalize(
		mean=feature_extractor.image_mean,
		std=feature_extractor.image_std
	)
	return norm


def get_train_transform(feature_extractor):
	''' Returns the transform that is applied to each training example '''
	norm = get_norm(feature_extractor)
	tform = transforms.Compose(
		[
			transforms.RandomResizedCrop(feature_extractor.size),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			norm,
		]
	)
	return tform


def get_val_transform(feature_extractor):
	''' Returns the transform that is applied to each validation (or testing) example '''
	norm = get_norm(feature_extractor)
	tform = transforms.Compose(
		[
			transforms.Resize(feature_extractor.size),
			transforms.CenterCrop(feature_extractor.size),
			transforms.ToTensor(),
			norm,
		]
	)
	return tform


def get_train_val_samplers(dataset, args):
	'''
	Gets training and validation set samplers based on an initial (training) dataset.
	Samplers should then be used in the respective training/validation dataloaders.
	If the test set is used as the validation set, there is no need to use samplers. 
	'''
	# If using the test set for validation, use a weighted sampler for the training set
	# and do not use a sampler for the validation set
	if args.test_for_val:
		# Don't use a weighted sampler if we are using a subset of the training data
		if args.train_val_size is not None:
			return None, None
		tgts = dataset.targets
		class_counts = np.array([len(np.where(tgts == t)[0]) for t in np.unique(tgts)])
		weight = 1. / class_counts
		samples_weight = np.array([weight[t] for t in tgts])

		samples_weight = torch.from_numpy(samples_weight)
		samples_weight = samples_weight.double()
		train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
		return train_sampler, None

	# Get the training and validation splits
	dataset_size = len(dataset)
	dataset_indices = list(range(dataset_size))
	split = int(np.floor((1 - args.percent_train) * dataset_size))
	# Randomly shuffle the dataset indices
	np.random.seed(args.seed)
	np.random.shuffle(dataset_indices)
	# Make samplers using train/validation splits of randomly shuffled indices
	train_sampler = SubsetRandomSampler(dataset_indices[split:])
	val_sampler = SubsetRandomSampler(dataset_indices[:split])
	return train_sampler, val_sampler
