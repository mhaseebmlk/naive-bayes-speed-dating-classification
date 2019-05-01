import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_FILE_NAME = 'dating.csv'
DATA_PATH = '../1_preprocessing/' + DATA_FILE_NAME

df = pd.read_csv(DATA_PATH)

female_filter = df['gender'].isin([0])
male_filter = df['gender'].isin([1])

df_female = df[female_filter]
df_male = df[male_filter]

preference_scores_of_participant_columns = [
	'attractive_important', 
	'sincere_important', 
	'intelligence_important',
	'funny_important', 
	'ambition_important', 
	'shared_interests_important'
]

means_female = dict()
means_male = dict()
for col in preference_scores_of_participant_columns:
	mean = round(df_female[col].mean(),2)
	means_female[col] = mean
for col in preference_scores_of_participant_columns:
	mean = round(df_male[col].mean(),2)
	means_male[col] = mean

# print ('means_female:', means_female)
# print ('means_male:',means_male)

preference_scores_of_participant_columns = [
	'attractive\nimportant', 
	'sincere\nimportant', 
	'intelligence\nimportant',
	'funny\nimportant', 
	'ambition\nimportant', 
	'shared\ninterests\nimportant'
]
OUTPUT_FILE_NAME='plot_2_1.png'
female_means = list(means_female.values())
male_means = list(means_male.values())
index = preference_scores_of_participant_columns
df = pd.DataFrame({'females': female_means,'males': male_means}, index=index)
ax = df.plot.bar(rot=0,color=['#ff7f0e', '#1f77b4'])
fig = ax.get_figure()
fig.savefig(OUTPUT_FILE_NAME)

print ('Bar plot saved in file {}'.format(OUTPUT_FILE_NAME))