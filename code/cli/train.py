import os
import math
import copy
import logging
import torch
import transformers
import wandb
import numpy as np

from tqdm import tqdm
from datasets import load_metric
from transformers import AutoFeatureExtractor, ResNetForImageClassification, Adafactor
from torch.utils.data import DataLoader

from emotion_detection.dataset_handler import FERDataset
from emotion_detection.utils import parse_args, get_train_transform, get_val_transform, get_train_val_samplers


# Setup/initialize logging
logger = logging.getLogger(__file__)
logging.basicConfig(
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
	datefmt="%m/%d/%Y %H:%M:%S",
	level=logging.INFO,
)
# Accuracy will be used as the main evaluation metric
accuracy = load_metric('accuracy')


def get_model_data(feature_extractor, args):
	'''
	Gets train/validation datasets and dataloaders, given a model's feature extractor
	Args:
		feature_extractor: The pre-trained models's feature extractor
		args: The set of arguments used to run this script
	'''
	# Get the training dataset
	train_data = FERDataset(
		args.data_dir,
		transform=get_train_transform(feature_extractor),
		split='train',
		sample_size=args.train_val_size
	)
	# Determine validation split and validation sample size based on if we are
	# using the test set or a subset of the training set as validation data
	val_split = args.test_type if args.test_for_val else 'val'
	val_sample_size = args.test_size if args.test_for_val else args.train_val_size
	# Get the validation dataset
	val_data = FERDataset(
		args.data_dir,
		transform=get_val_transform(feature_extractor),
		split=val_split,
		sample_size=val_sample_size
	)
	# Get training and validation samplers based on if we are using the test set
	# or a subset of the training set as validation data. If we are using the test set
	# as the validation set, there is no sampler so we need to shuffle training data
	train_sampler, val_sampler = get_train_val_samplers(train_data, args)
	shuffle_train = train_sampler is None
	# Get the training and validation dataloaders
	train_dataloader = DataLoader(
		train_data,
		batch_size=args.batch_size,
		sampler=train_sampler,
		shuffle=shuffle_train
	)
	val_dataloader = DataLoader(
		val_data,
		batch_size=args.batch_size,
		sampler=val_sampler,
		shuffle=False
	)

	dsets = {'train': train_data, 'validation': val_data}
	dloaders = {'train': train_dataloader, 'validation': val_dataloader}
	return dsets, dloaders


def evaluate_model(model, dataloader, args):
	'''
	Evaluate a model on a dataloader
	Args:
		model: The model to evaluate
		dataloader: The dataloader to use for evaluation data
		args: The set of arguments used to run this script
	'''
	model.eval()
	loss = 0.0

	eval_labels, eval_preds = [], []
	for inputs, labels in tqdm(dataloader, desc='Evaluation'):
		with torch.inference_mode():
			inputs = inputs.to(args.device)
			labels = labels.to(args.device)
			
			model_output = model(pixel_values=inputs, labels=labels)
			logits = model_output.logits.to(args.device)
			loss += model_output.loss.to(args.device)

			preds = logits.argmax(-1).to(args.device)
			accuracy.add_batch(predictions=preds, references=labels)

			eval_labels += labels.tolist()
			eval_preds += preds.tolist()

	model.train()
	eval_accuracy = accuracy.compute()['accuracy']
	eval_loss = loss / len(dataloader.dataset)
	return eval_accuracy, eval_loss, eval_labels, eval_preds


def train_model(model, train_dataloader, eval_dataloader, optimizer, scheduler, eval_dataset, args):
	'''
	Train (fine-tune) a pre-trained model
	Args:
		model: The model to train
		train_dataloader: The dataloader to use for training data
		eval_dataloader: The dataloader to use for evaluation data
		optimizer: The optimizer to use for training
		scheduler: The learning rate scheduler to use for training
		args: The set of arguments used to run this training script
	'''
	# Keep track of the best model weights and the best validation accuracy
	best_weights = copy.deepcopy(model.state_dict())
	best_eval_accuracy = 0.0

	progress_bar = tqdm(range(args.max_train_steps))
	global_step = 0
	# Train for the specified number of epochs
	for epoch in range(args.num_train_epochs):
		# Set model to training mode, process each batch in the train dataloader
		model.train()
		for inputs, labels in train_dataloader:
			# Move batch of inputs and labels to the specified device
			inputs = inputs.to(args.device)
			labels = labels.to(args.device)
			# Get model outputs given the current batch of inputs and labels
			model_output = model(pixel_values=inputs, labels=labels)
			# Zero out (clear) the gradients before backpropagation
			optimizer.zero_grad()
			# Get the model's loss for current batch, then perform backpropagation
			loss = model_output.loss.to(args.device)
			loss.backward()
			# Advance the optimizer and lr scheduler
			optimizer.step()
			if scheduler is not None:
				scheduler.step()

			progress_bar.update(1)
			global_step += 1
			if args.use_wandb:
				wandb.log(
					{
						'train_loss': loss,
						'learning_rate': optimizer.param_groups[0]['lr'],
						'epoch': epoch,
					},
					step=global_step,
				)

			# Logging step count reached, log our batch-level train accuracy and loss
			if global_step % args.logging_steps == 0:
				# Get logits (class probs) and predictions (class w/ highest prob)
				logits = model_output.logits.to(args.device)
				preds = logits.argmax(-1).to(args.device)
				# Compute batch-level train accuracy 
				train_accuracy = accuracy.compute(predictions=preds, references=labels)['accuracy']
				logger.info(
					f'\n(Epoch={epoch}, Step={global_step}, Batch Size={len(inputs)})'
					f' - Train Batch Accuracy: {train_accuracy}, Train Loss: {loss}'
				)
				if args.use_wandb:
					wandb.log(
						{'train_accuracy': train_accuracy},
						step=global_step,
					)
				
			# Evaluation step count reached, log our validation set accuracy and loss
			if global_step % args.eval_every_steps == 0 or global_step == args.max_train_steps:
				# Evaluate the current model using the evaluation dataloader
				eval_accuracy, eval_loss, eval_labels, eval_preds = evaluate_model(
					model=model,
					dataloader=eval_dataloader,
					args=args
				)
				# Check if we have a new best model and save its weights if so
				if eval_accuracy > best_eval_accuracy:
					best_eval_accuracy = eval_accuracy
					best_weights = copy.deepcopy(model.state_dict())
				
				logger.info(
					f'\n(Epoch={epoch + 1}, Step={global_step},'
					f' #Val Samples={len(eval_dataloader.dataset)})'
					f' - Val Accuracy: {eval_accuracy}, Val Loss: {eval_loss}'
				)
				if args.use_wandb:
					# Log eval accuracy and loss
					wandb.log(
						{'eval_accuracy': eval_accuracy, 'eval_loss': eval_loss},
						step=global_step,
					)
					# Log eval cofusion matrix if eval accuracy improved
					if eval_accuracy == best_eval_accuracy:
						wandb.log(
							{
								'conf_mat': wandb.plot.confusion_matrix(
									y_true=eval_labels,
									preds=eval_dataset.indices_to_classes(eval_preds),
									class_names=list(eval_dataset.CLASSES.values())
								)
							}
						)

			# Checkpoing step count reached, save the current model checkpoint
			if global_step % args.checkpoint_every_steps == 0:
				logger.info(
					f'\n(Epoch={epoch}, Step={global_step})'
					f' - Saving model checkpoint to: {args.output_dir}'
				)
				model.save_pretrained(args.output_dir)

			# Stop training if we reach the pre-defined max # of training steps
			if global_step >= args.max_train_steps:
				break

	# Load the model with the best weights we found and return it
	model.load_state_dict(best_weights)
	return model, best_eval_accuracy


def main():
	# Parse training arguments	
	args = parse_args()
	logger.info(f'Starting script with arguments: {args}')

	# Initialize wandb as soon as possible to log all stdout to the cloud
	if args.use_wandb:
		wandb.init(project=args.wandb_project, config=args)
	# Make sure output directory exists, if not create it
	os.makedirs(args.output_dir, exist_ok=True)

	# Get the pre-trained feature extractor and model
	feature_extractor = AutoFeatureExtractor.from_pretrained(args.pretrained_model_name)
	model = ResNetForImageClassification.from_pretrained(args.pretrained_model_name)
	# Move the model to the specified device for training
	model = model.to(device=args.device)
	if args.use_wandb:
		wandb.watch(model)

	# If in debug mode, only use a small, pre-defined subset for training
	if args.debug:
		args.train_val_size = 75
		args.test_size = 20
		args.percent_train = 1.0

	# Get the dataloaders (training/validation/testing) for the FER-2013 dataset
	dsets, dloaders = get_model_data(feature_extractor, args)
	train_dataset = dsets['train']
	eval_dataset = dsets['validation']
	train_dataloader = dloaders['train']
	eval_dataloader = dloaders['validation']

	# If in debug mode, use training set as eval set (to check if we can overfit)
	if args.debug:
		eval_dataloader = train_dataloader

	# Scheduler and math around the number of training steps
	num_update_steps_per_epoch = len(train_dataloader)
	# If there is no max # of training steps (None by default), set it based on
	# the # of training epochs and update steps per epoch. Otherwise, use the
	# max # of training steps to set the # of training epochs.
	if args.max_train_steps is None:
		args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
	else:
		args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

	# Optimizer to use during training
	optimizer = Adafactor(
		model.parameters(),
		scale_parameter=False,
		relative_step=False,
		warmup_init=False,
		lr=args.learning_rate,
		weight_decay=args.weight_decay
	)
	# transformers.SchedulerType
	if args.lr_scheduler_type == 'no_scheduler':
		scheduler = None
	else:
		# Scheduler to use during training
		scheduler = transformers.get_scheduler(
			name=args.lr_scheduler_type,
			optimizer=optimizer,
			num_warmup_steps=args.num_warmup_steps,
			num_training_steps=args.max_train_steps
		)

	logger.info('***** Running training *****')
	logger.info(f'  Num examples = {len(train_dataloader)}')
	logger.info(f'  Num Epochs = {args.num_train_epochs}')
	logger.info(f'  Total optimization steps = {args.max_train_steps}')

	# Fine-tune our pre-trained model
	finetuned_model, best_eval_accuracy = train_model(
		model=model,
		train_dataloader=train_dataloader,
		eval_dataloader=eval_dataloader,
		optimizer=optimizer,
		scheduler=scheduler,
		eval_dataset=eval_dataset,
		args=args
	)

	# Save the fine-tuned model with the best weights
	logger.info(f'Saving final (best) model checkpoint to {args.output_dir}')
	finetuned_model.save_pretrained(args.output_dir)

	# If using wandb, save the model and its config to wandb
	if args.use_wandb:
		logger.info('Uploading model and config to wandb')
		# wandb.run.summary['eval_accuracy'] = best_eval_accuracy
		wandb.save(os.path.join(args.output_dir, '*'))

	logger.info(f'Script finished succesfully, model saved in {args.output_dir}')


if __name__ == '__main__':
	main()
