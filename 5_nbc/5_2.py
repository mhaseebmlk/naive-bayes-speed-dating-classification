import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_FILE_NAME = 'dating.csv'
DATA_PATH = '../1_preprocessing/' + DATA_FILE_NAME

BIN_SIZES = [2, 5, 10, 50, 100, 200]
RANDOM_STATE = 47
FRAC = 0.2

DATA_COLUMNS = ['gender', 'age', 'age_o', 'race', 'race_o', 'samerace', 'importance_same_race', 'importance_same_religion', 'field', 'pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests', 'attractive_important', 'sincere_important', 'intelligence_important', 'funny_important', 'ambition_important', 'shared_interests_important', 'attractive', 'sincere', 'intelligence', 'funny', 'ambition', 'attractive_partner', 'sincere_partner', 'intelligence_parter', 'funny_partner', 'ambition_partner', 'shared_interests_partner', 'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga', 'interests_correlate', 'expected_happy_with_sd_people', 'like']

label_class_name = 'decision'

continuous_valued_columns = ['age', 'age_o', 'importance_same_race', 'importance_same_religion', 'pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests', 'attractive_important', 'sincere_important', 'intelligence_important', 'funny_important', 'ambition_important', 'shared_interests_important', 'attractive', 'sincere', 'intelligence', 'funny', 'ambition', 'attractive_partner', 'sincere_partner', 'intelligence_parter', 'funny_partner', 'ambition_partner', 'shared_interests_partner', 'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga', 'interests_correlate', 'expected_happy_with_sd_people', 'like']

# continuous valued column ranges as mentioend in the field-meaning.pdf. both values are inclusive
col_ranges = {'age': (18, 58), 'age_o': (18, 58), 'importance_same_race': (0, 10), 'importance_same_religion': (0, 10), 'pref_o_attractive': (0, 1), 'pref_o_sincere': (0, 1), 'pref_o_intelligence': (0, 1), 'pref_o_funny': (0, 1), 'pref_o_ambitious': (0, 1), 'pref_o_shared_interests': (0, 1), 'attractive_important': (0, 1), 'sincere_important': (0, 1), 'intelligence_important': (0, 1), 'funny_important': (0, 1), 'ambition_important': (0, 1), 'shared_interests_important': (0, 1), 'attractive': (0, 10), 'sincere': (0, 10), 'intelligence': (0, 10), 'funny': (0, 10), 'ambition': (0, 10), 'attractive_partner': (0, 10), 'sincere_partner': (0, 10), 'intelligence_parter': (0, 10), 'funny_partner': (0, 10), 'ambition_partner': (0, 10), 'shared_interests_partner': (0, 10), 'sports': (0, 10), 'tvsports': (0, 10), 'exercise': (0, 10), 'dining': (0, 10), 'museums': (0, 10), 'art': (0, 10), 'hiking': (0, 10), 'gaming': (0, 10), 'clubbing': (0, 10), 'reading': (0, 10), 'tv': (0, 10), 'theater': (0, 10), 'movies': (0, 10), 'concerts': (0, 10), 'music': (0, 10), 'shopping': (0, 10), 'yoga': (0, 10), 'interests_correlate': (-1, 1), 'expected_happy_with_sd_people': (0, 10), 'like': (0, 10)}

def update_anomalous_max_values(row, col_min, col_max , true_range):
	"""
	Will update a cell having a value greater than the max value mentioned in the docs with the max value mentioned in the docs.

	row := one row of the df
	col_min := the minimum value of the column within this dataset
	col_max := the maximum value of the column within this dataset
	range := a tuple representing the range of the column. Both sides are inclusive.
	"""
	true_max = true_range[1]
	row = true_max if row > true_max else row
	return row

def bin_row(row, intervals):
	bin_num = intervals.index(row)
	return bin_num

def discretize(df, num_bins, output_file_name=None, print_=True):
	bin_interval_mappings = dict()
	for col in continuous_valued_columns:
		col_min, col_max = df[col].min(), df[col].max()
		true_range = col_ranges[col]
		df[col] = df[col].apply(update_anomalous_max_values, args=(col_min,col_max, true_range ))

		true_min, true_max = true_range[0], true_range[1]
		bin_min_val = -0.01 if true_min==0 else (true_min-(0.1/100 * true_min))
		# bin_max_val = 0.01 if true_max==0 else (true_max+(0.1/100 * true_max))

		bins = list(np.linspace(true_min, true_max, num_bins+1))
		bins = list(zip(bins, bins[1:]))
		bins[0] = (bin_min_val,bins[0][1])
		bins = pd.IntervalIndex.from_tuples(bins)

		bin_interval_mappings[col] = {bin: interval for bin, interval in enumerate(bins)}

		bins_list = bins.tolist()
		df[col] = pd.cut(df[col], bins).apply(bin_row, intervals= bins_list)

		counts = [len(df[df[col] == i])  for i in range(len(bins_list))]

		if print_==True:
			print ('{}: {}'.format(col, counts))

	# if print_ == True:
	# 	for k in bin_interval_mappings.keys(): 
	# 		print ('col = {}, mappings: {}'.format(k,bin_interval_mappings[k]))

	if output_file_name != None:
		df.to_csv(output_file_name, index=False)

	return df

def predict(row, class_labels, nbc_params):
	probs = []
	for i in class_labels:
		prior_prob = nbc_params['prior_probs'][i]
		prob = prior_prob
		for col in DATA_COLUMNS:
			key = '{}={}|{}={}'.format(col,row[col],label_class_name,i)
			# conditional_prob = nbc_params['conditional_probs'][key]
			conditional_prob = 0
			try:
				conditional_prob = nbc_params['conditional_probs'][key]
			except:
				pass
			prob = prob * conditional_prob
		probs.append((i,prob))

	assigned_label = -1
	max_prob = -1
	for p in probs:
		if p[1] > max_prob:
			max_prob = p[1]
			assigned_label = p[0]

	return 'Correct' if assigned_label==row[label_class_name] else 'Incorrect'

def nbc(df_training_org, df_testing,t_frac, debug=False, print_=True):
	df_training = df_training_org.sample(frac=t_frac, random_state=RANDOM_STATE)

	# calculate the prior probs of the class labels
	parameters=dict()
	prior_probs = dict()
	class_labels = df_training[label_class_name].unique()
	for v in class_labels:
		label_filter = df_training[label_class_name].isin([v])
		filtered_df = df_training[label_filter]
		prob = len(filtered_df)/len(df_training)
		prior_probs[v] = prob

	parameters['prior_probs'] = prior_probs

	# get the naive bayes probabilities for each class label
	conditional_probs = dict()
	for i in class_labels:
		for col in DATA_COLUMNS:
			col_vals = df_training[col].unique()
			for j in col_vals:
				key = '{}={}|{}={}'.format(col,j,label_class_name,i)
				col_val_filter = df_training[col].isin([j])
				label_filter = df_training[label_class_name].isin([i])
				filtered_df = df_training[col_val_filter & label_filter]
				label_counts = len(df_training[label_filter])
				prob = (len(filtered_df))/(label_counts)
				# prob = (len(filtered_df)+1)/(label_counts + len(col_vals)) # getting .1% better accuracy without smoothing
				conditional_probs[key] = prob

	parameters['conditional_probs'] = conditional_probs
	if debug:
		print (parameters)

	# model has been learned. now get accuracies
	train_accuracy = test_accuracy = 0.0
	# training accuracy
	df_training['predictions'] = df_training.apply(predict, axis=1, args=(class_labels,parameters))
	# print (df_training['predictions'].value_counts().values.tolist())
	prediction_counts = df_training['predictions'].value_counts().values.tolist()
	num_correct_predictions = prediction_counts[0]
	train_accuracy = num_correct_predictions/(sum(prediction_counts))*100
	# testing accuracy
	df_testing['predictions'] = df_testing.apply(predict, axis=1, args=(class_labels,parameters))
	# print (df_testing['predictions'].value_counts().values.tolist())
	prediction_counts = df_testing['predictions'].value_counts().values.tolist()
	num_correct_predictions = prediction_counts[0]
	test_accuracy = num_correct_predictions/(sum(prediction_counts))*100

	if print_:
		print ('Training Accuracy: {}'.format(train_accuracy))
		print ('Testing Accuracy: {}'.format(test_accuracy))

	return (train_accuracy, test_accuracy)

def main():

	training_accuracies = []
	test_accuracies = []
	for b in BIN_SIZES:

		df = pd.read_csv(DATA_PATH)

		print ('Bin size: {}'.format(b))

		# binning
		df_binned = discretize(df, b, output_file_name=None, print_=False)

		# train/test split
		df_test = df_binned.sample(frac=FRAC, random_state=RANDOM_STATE)
		df_train = df_binned.sample(frac=(1-FRAC), random_state=RANDOM_STATE)

		t_frac = 1
		train_acc, test_acc = nbc(df_train, df_test, t_frac, debug=False, print_=True)

		training_accuracies.append(train_acc)
		test_accuracies.append(test_acc)

	# print (training_accuracies, test_accuracies)
	# print (BIN_SIZES)

	plot_name = 'plot_5_2.png'

	fig, ax = plt.subplots()
	line1, = ax.plot(BIN_SIZES, training_accuracies, label='train')
	line1.set_dashes([2, 2, 10, 2]) 
	line2, = ax.plot(BIN_SIZES, test_accuracies, label='test')
	ax.legend()
	title='Training and Test Accuracies with different number of Bins'
	plt.xlabel('Number of bins')
	plt.ylabel('Accuracy')
	plt.title(title)
	plt.savefig(plot_name)

if __name__ == '__main__':
	main()