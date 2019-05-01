import pandas as pd
import numpy as np

DATA_FILE_NAME = 'dating-binned.csv'
DATA_PATH = '../3_discretization/' + DATA_FILE_NAME
OUTPUT_FILE_NAME_TRAINING = 'trainingSet.csv'
OUTPUT_FILE_NAME_TEST = 'testSet.csv'

RANDOM_STATE = 47
FRAC = 0.2

def split(df, FRAC, RANDOM_STATE):
	df_test = df.sample(frac=FRAC, random_state=RANDOM_STATE)
	df_train = df.sample(frac=(1-FRAC), random_state=RANDOM_STATE)

	df_test.to_csv(OUTPUT_FILE_NAME_TEST, index=False)
	df_train.to_csv(OUTPUT_FILE_NAME_TRAINING, index=False)

def main():
	df = pd.read_csv(DATA_PATH)
	split(df, FRAC, RANDOM_STATE)

if __name__ == '__main__':
	main()