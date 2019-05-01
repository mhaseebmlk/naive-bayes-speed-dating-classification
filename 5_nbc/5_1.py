import pandas as pd
import numpy as np

TRAINING_DATE_FILE_NAME = 'trainingSet.csv'
TRAINING_DATE_PATH = '../4_splitting/' + TRAINING_DATE_FILE_NAME
TESTING_DATE_FILE_NAME = 'testSet.csv'
TESTING_DATE_PATH = '../4_splitting/' + TESTING_DATE_FILE_NAME

RANDOM_STATE = 47

DATA_COLUMNS = ['gender', 'age', 'age_o', 'race', 'race_o', 'samerace', 'importance_same_race', 'importance_same_religion', 'field', 'pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests', 'attractive_important', 'sincere_important', 'intelligence_important', 'funny_important', 'ambition_important', 'shared_interests_important', 'attractive', 'sincere', 'intelligence', 'funny', 'ambition', 'attractive_partner', 'sincere_partner', 'intelligence_parter', 'funny_partner', 'ambition_partner', 'shared_interests_partner', 'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga', 'interests_correlate', 'expected_happy_with_sd_people', 'like']

label_class_name = 'decision'

def predict(row, class_labels, nbc_params):
	probs = []
	for i in class_labels:
		prior_prob = nbc_params['prior_probs'][i]
		prob = prior_prob
		for col in DATA_COLUMNS:
		# for col in ['clubbing']:	
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

def nbc(t_frac, debug=False, print_=True):
	df_training_org = pd.read_csv(TRAINING_DATE_PATH)
	df_training = df_training_org.sample(frac=t_frac, random_state=RANDOM_STATE)
	df_testing = pd.read_csv(TESTING_DATE_PATH)

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
		# for col in ['clubbing']:
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
	# print (df_training['predictions'].value_counts())
	prediction_counts = df_training['predictions'].value_counts().values.tolist()
	num_correct_predictions = prediction_counts[0]
	train_accuracy = num_correct_predictions/(sum(prediction_counts))*100
	# testing accuracy
	df_testing['predictions'] = df_testing.apply(predict, axis=1, args=(class_labels,parameters))
	# print (df_testing['predictions'].value_counts())
	prediction_counts = df_testing['predictions'].value_counts().values.tolist()
	num_correct_predictions = prediction_counts[0]
	test_accuracy = num_correct_predictions/(sum(prediction_counts))*100

	if print_:
		print ('Training Accuracy: {}'.format(train_accuracy))
		print ('Testing Accuracy: {}'.format(test_accuracy))


def main():
	t_frac = 1
	nbc(t_frac, debug=False, print_=True)

if __name__ == '__main__':
	main()