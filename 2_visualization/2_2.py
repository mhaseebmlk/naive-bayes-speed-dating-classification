import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_FILE_NAME = 'dating.csv'
DATA_PATH = '../1_preprocessing/' + DATA_FILE_NAME

df = pd.read_csv(DATA_PATH)

rating_of_partner_from_participant_columns =  [
	'attractive_partner', 
	'sincere_partner', 
	'intelligence_parter',
	'funny_partner', 
	'ambition_partner', 
	'shared_interests_partner'
]

distinct_vals = list()
for col in rating_of_partner_from_participant_columns:
	distinct_vals.append( (col,sorted(df[col].unique())) )

# compute success rates
success_rates = dict()
for col, vals in distinct_vals:
	success_rates[col] = dict()
	for val in vals:
		filter_val = df[col].isin([val])
		filter_decision = df['decision'].isin([1])
		num_val = len(df[filter_val])
		num_success = len(df[filter_val & filter_decision])
		success_rate = round(num_success/num_val,2)
		success_rates[col][val] = success_rate
		
	# print ('Column: {}\n\tvalues: {}\n\trates: {}\n'.format(col,vals,list(success_rates[col].values())))

	plot_name = 'plot_2_2_{}.png'.format(col)
	plt.figure()
	plt.scatter(vals, list(success_rates[col].values()))
	title='Success Rate v.s. Rating of partner on {}'.format(col)
	plt.xlabel('Values')
	plt.ylabel('Success Rate')
	plt.title(title)
	plt.xticks(range(11))
	plt.savefig(plot_name)