import pandas as pd

'''
Script used to extract and format the test examples located in Kaggle's "fer2013.csv" dataset,
which can be found in the following .tar.gz file:
	- https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=fer2013.tar.gz

This is so that the testing examples have the same format as the "train.csv" file found here:
	https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=train.csv

The testing set that is not located in the .tar.gz file has no class labels,
so it cannot be used for validation purposes
'''


# File names and paths
full_fname = 'fer2013.csv'
test_fname = 'test.csv'
test_path = '/Users/christopherlewis/Desktop/Deep Learning/Final Project/Code/dataset/fer2013/'

# Read and format the test data and write it to "test.csv"
full_df = pd.read_csv(full_fname)
test_df = full_df.loc[full_df['Usage'] == 'PublicTest'].drop('Usage', axis=1)
test_df.to_csv(test_path + test_fname, index=False)
