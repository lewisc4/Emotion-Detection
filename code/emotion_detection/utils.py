import argparse
import torch
import numpy as np

from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


def parse_args():
	"""This function creates argument parser and parses the scrip input arguments.
	This is the most common way to define input arguments in python. It is used
	by train.py and human_eval.py

	To change the parameters, pass them to the script, for example:

	python cli/train.py \
		--output_dir output_dir \
		--weight_decay 0.01
	
	Default arguments have the meaning of being a reasonable default value, not of the last arguments used.
	"""
	parser = argparse.ArgumentParser(description="Train machine translation transformer model")

	# Required arguments
	parser.add_argument(
		"--data_dir",
		type=str,
		default="dataset",
		help="Where the dataset is stored.",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default="outputs",
		help="Where to store the final model.",
	)
	parser.add_argument(
		"--pretrained_model_name",
		type=str,
		default="microsoft/resnet-18",
		help="Name of pretrained model to be used.",
	)
	parser.add_argument(
		"--dataset_dir",
		type=str,
		default="dataset",
		help="Directory where the dataset is stored.",
	)
	parser.add_argument(
		"--test_for_val",
		default=False,
		action="store_true",
		help="Use the test set for validation. By default, a subset of training data is used.",
	)
	parser.add_argument(
		"--train_val_size",
		type=int,
		default=None,
		help="Combined size (# samples) of the training and validation set.",
	)
	parser.add_argument(
		"--test_size",
		type=int,
		default=None,
		help="Size (# samples) of the test set.",
	)
	parser.add_argument(
		"--percent_train",
		type=float,
		default=0.8,
		help="Percentage of the data to use for training (train_val_size * percent_train).",
	)
	parser.add_argument(
		"--debug",
		default=False,
		action="store_true",
		help="Whether to use a small subset of the dataset for debugging.",
	)

	# Training arguments
	parser.add_argument(
		"--device",
		default="cuda" if torch.cuda.is_available() else "cpu",
		help="Device (cuda or cpu) on which the code should run",
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=128,
		help="Batch size (per device) for the training dataloader.",
	)
	parser.add_argument(
		"--learning_rate",
		type=float,
		default=5e-4,
		help="Initial learning rate (after the potential warmup period) to use.",
	)
	parser.add_argument(
		"--weight_decay",
		type=float,
		default=0.0,
		help="Weight decay to use.",
	)
	parser.add_argument(
		"--num_train_epochs",
		type=int,
		default=15,
		help="Total number of training epochs to perform.",
	)
	parser.add_argument(
		"--eval_every_steps",
		type=int,
		default=40,
		help="Perform evaluation every n network updates.",
	)
	parser.add_argument(
		"--logging_steps",
		type=int,
		default=20,
		help="Compute and log training batch metrics every n steps.",
	)
	parser.add_argument(
		"--checkpoint_every_steps",
		type=int,
		default=500,
		help="Save model checkpoint every n steps.",
	)
	parser.add_argument(
		"--max_train_steps",
		type=int,
		default=None,
		help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
	)
	parser.add_argument(
		"--lr_scheduler_type",
		type=str,
		default="linear",
		help="The scheduler type to use.",
		choices=["no_scheduler", "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
	)
	parser.add_argument(
		"--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="A seed for reproducible training.",
	)
	
	# Weights and biases (wandb) arguments
	parser.add_argument(
		"--use_wandb",
		default=True,
		action="store_true",
		help="Whether to enable usage/logging for the wandb_project.",
	)
	parser.add_argument(
		"--wandb_project", 
		default="emotion_detection",
		help="wandb project name to log metrics to"
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
	# If using the test set for validation, do no samplers are needed
	if args.test_for_val:
		return None, None
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
