import os
import errno
import logging
import argparse
import pandas as pd

from pathlib import Path


"""
Script used to extract and format the test examples located in Kaggle's "fer2013.csv" dataset,
which can be found in the following .tar.gz file:
	- https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=fer2013.tar.gz

This is so that the testing examples have the same format as the "train.csv" file found here:
	https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=train.csv

The testing set that is not located in the .tar.gz file has no class labels,
so it cannot be used for validation purposes
"""


# Setup/initialize logging
logger = logging.getLogger(__file__)
logging.basicConfig(
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
	datefmt="%m/%d/%Y %H:%M:%S",
	level=logging.INFO,
)


def parse_args():
	parser = argparse.ArgumentParser(description="Prepare dataset(s) for image classification")
	# Script cli arguments
	parser.add_argument(
		"--input_data_dir",
		type=str,
		default="fer2013",
		help="Where the initial, unprocessed dataset is stored."
	)
	parser.add_argument(
		"--output_data_dir",
		type=str,
		default="dataset/fer2013",
		help="Where to save the processed dataset."
	)
	# Parse and return arguments
	args = parser.parse_args()
	return args


def make_dataset(from_df, usage, save_location):
	""" Makes a dataset from a df, based on usage. Saves to specificed location. """
	# Don't make the dataset if it already exists
	if Path(save_location).exists():
		return
	logger.info(f"Making: {save_location}...")
	# Get the data for the specified usage and drop the usage column
	dataset = from_df.loc[from_df["Usage"] == usage].drop("Usage", axis=1)
	# Save the dataset to the specified location and do not write the df index
	dataset.to_csv(save_location, index=False)


def main():
	# Get the cli script arguments
	args = parse_args()

	# The file name and path of the initial, unprocessed dataset
	full_dataset_file = os.path.join(args.input_data_dir, "fer2013.csv")
	full_dataset_path = Path(full_dataset_file)

	# Check if the input data directory exists, raise error if it does not
	if full_dataset_path.exists():
		logger.info(f"Reading input file: {full_dataset_file}...")
		full_dataset_df = pd.read_csv(full_dataset_file)
	else:
		raise FileNotFoundError(
			errno.ENOENT, os.strerror(errno.ENOENT, full_dataset_path)
		)
	# Make sure the dataset directory exists, create it if not
	os.makedirs(args.output_data_dir, exist_ok=True)

	# The file name to use for the training dataset
	train_file = os.path.join(args.output_data_dir, "train.csv")
	# The file names to use for the public/private test datasets
	public_test_file = os.path.join(args.output_data_dir, "public_test.csv")
	private_test_file = os.path.join(args.output_data_dir, "private_test.csv")

	# Make the training dataset
	make_dataset(
		from_df=full_dataset_df,
		usage="Training",
		save_location=train_file
	)
	# Make the public/private test datasets
	make_dataset(
		from_df=full_dataset_df,
		usage="PublicTest",
		save_location=public_test_file
	)
	make_dataset(
		from_df=full_dataset_df,
		usage="PrivateTest",
		save_location=private_test_file
	)


if __name__ == "__main__":
	main()
