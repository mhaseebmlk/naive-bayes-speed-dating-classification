import sys

import pandas as pd
import numpy as np

if len(sys.argv) != 3:
	print('Usage: \n\tpython3 preprocess.py dating-full.csv dating.csv')
	exit(1)

DATA_FILE_NAME = sys.argv[1]
DATA_PATH = '../dataset/' + DATA_FILE_NAME
OUTPUT_FILE_NAME = sys.argv[2]

df = pd.read_csv(DATA_PATH)

preference_scores_of_participant_columns = [
	'attractive_important', 
	'sincere_important', 
	'intelligence_important',
	'funny_important', 
	'ambition_important', 
	'shared_interests_important'
]
for col in preference_scores_of_participant_columns:
	df[col] = df[col].astype('float64')

preference_scores_of_partner_columns = [
	'pref_o_attractive', 
	'pref_o_sincere',
	'pref_o_intelligence', 
	'pref_o_funny',
	'pref_o_ambitious', 
	'pref_o_shared_interests'
]

discrete_valued_columns = [
	'gender', 'race', 'race_o', 
	'samerace', 'field', 'decision'
]

continuous_valued_columns = ['age', 'age_o', 'importance_same_race', 'importance_same_religion', 'pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests', 'attractive_important', 'sincere_important', 'intelligence_important', 'funny_important', 'ambition_important', 'shared_interests_important', 'attractive', 'sincere', 'intelligence', 'funny', 'ambition', 'attractive_partner', 'sincere_partner', 'intelligence_parter', 'funny_partner', 'ambition_partner', 'shared_interests_partner', 'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga', 'interests_correlate', 'expected_happy_with_sd_people', 'like']

rating_of_partner_from_participant_columns =  [
	'attractive_partner', 
	'sincere partner', 
	'intelligence_partner',
	'funny_partner', 
	'ambition_partner', 
	'shared_interests_partner'
]

count_quotes_removed = count_standardized = 0
label_encoding_dict = dict()

cols_to_split_quotes = ['race', 'race_o','field']
cols_to_standardize = ['field']
cols_to_label_encode = ['gender', 'race', 'race_o', 'field']

def strip_quotes(row):
	global count_quotes_removed
	stripped_row = row
	if row[0] == '\'' or  row[-1] == '\'':
		row = row.strip('\'')
		count_quotes_removed += 1
	return row

def standardize(row):
	global count_standardized
	if not row.islower():
		row = row.lower()
		count_standardized += 1
	return row

for col in cols_to_split_quotes:
	df[col] = df[col].apply(strip_quotes)

for col in cols_to_standardize:
	df[col] = df[col].apply(standardize)

for i, row in df.iterrows():
	total_pref_scores_participant = total_pref_scores_partner = 0.0
	for col in preference_scores_of_participant_columns:
		total_pref_scores_participant += row[col]	
	for col in preference_scores_of_participant_columns:
		df.at[i, col] = row[col]/total_pref_scores_participant

	for col in preference_scores_of_partner_columns:
		total_pref_scores_partner += df.at[i, col]
	for col in preference_scores_of_partner_columns:
		df.at[i, col] = row[col]/total_pref_scores_partner

print ('Quotes removed from {} cells.'.format(count_quotes_removed))
print ('Standardized {} cells to lower case.'.format(count_standardized))

# label encoding
for col in cols_to_label_encode:
	df[col] = df[col].astype('category')
	label_encoding_dict[col] = dict(enumerate(df[col].cat.categories))
	df[col] = df[col].cat.codes

# print(label_encoding_dict)

col_values_for_mappings = {'gender': 'male', 'race': 'European/Caucasian-American', 'race_o': 'Latino/Hispanic American', 'field': 'law'}
for col in cols_to_label_encode:
	mappings = label_encoding_dict[col]
	for encoded, decoded in mappings.items():
		if decoded == col_values_for_mappings[col]:
			print ('Value assigned for {} in column {}: {}.'.format(col_values_for_mappings[col], col, encoded))


# print the means of the columns
for col in preference_scores_of_participant_columns:
	mean = round(df[col].mean(),2)
	print('Mean of {}: {}.'.format(col,mean))
for col in preference_scores_of_partner_columns:
	mean = round(df[col].mean(),2)
	print('Mean of {}: {}.'.format(col,mean))

# save the file
df.to_csv(OUTPUT_FILE_NAME, index=False)
