#!/Users/christine/anaconda3/bin/python
# -*- coding: utf-8 -*-

# create_data_dir_CV.py
# create data directory for training BERT model using cross validation
# save formatted data folder as bertdata/bertdata_*

import os, sys, pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import KFold

os.chdir('data')

TEXT_COLUMN = sys.argv[1] # use 'abstract' or 'title' or 'title+abstract' as training data
DATASET = sys.argv[2] # which annotations dataset to use ('1' or '2')
NUM_FOLDS = int(sys.argv[3]) # number of folds for cross val

ANNOTATIONS_FILE = 'annotations{}.csv'.format(DATASET)
ARTICLE_INFO = 'article_info{}.csv'.format(DATASET)
OUTPUT_FOLDER = '../bertdata/bertdata{}_{}_CV'.format(DATASET, TEXT_COLUMN)

TRAIN_TEST_SPLIT = 0.8
tokenizer = RegexpTokenizer(r'\w+')

print('input type:', TEXT_COLUMN)
print('batch:', DATASET)
print('annotations file:', ANNOTATIONS_FILE)
print('article info:', ARTICLE_INFO)
print('output folder:', OUTPUT_FOLDER)

# create output folders
try: os.system('rm -rf '+OUTPUT_FOLDER) # delete folder if already there
except: pass
os.mkdir(OUTPUT_FOLDER)
for i in range(NUM_FOLDS):
	OUTPUT_FOLDER_FOLD = f'{OUTPUT_FOLDER}/{i}'
	os.mkdir(OUTPUT_FOLDER_FOLD)
	os.mkdir(OUTPUT_FOLDER_FOLD+'/train')
	os.mkdir(OUTPUT_FOLDER_FOLD+'/test')
	os.mkdir(OUTPUT_FOLDER_FOLD+'/train/relevant')
	os.mkdir(OUTPUT_FOLDER_FOLD+'/train/irrelevant')
	os.mkdir(OUTPUT_FOLDER_FOLD+'/test/relevant')
	os.mkdir(OUTPUT_FOLDER_FOLD+'/test/irrelevant')

# customizing the nltk English stop list
stoplist = stopwords.words('english')
stoplist.extend(["ever", "one", "do","does","make", "go", "us", "to", "get", "about", "may", "s", ".", ",", "!", "i", "I", '\"', "?", ";", "--", "--", "would", "could", "”", "Mr.", "Miss", "Mrs.", "don’t", "said", "can't", "didn't", "aren't", "I'm", "you're", "they're", "'s"])

# read in labels
df = pd.read_csv(ANNOTATIONS_FILE)
df = df[df['relevance'] < 2]
labels = {int(row['pubmed_id']):int(row['relevance']) for i,row in df.iterrows()}
THRESHOLD = int(len(labels) * TRAIN_TEST_SPLIT) # define threshold using train_test_split
print(len(labels), THRESHOLD)

# read in abstracts and titles
df = pd.read_csv(ARTICLE_INFO)
df = df.dropna(subset=['pubmed_id'])
df['pubmed_id'] = df['pubmed_id'].astype('int')
df = df[df['pubmed_id'].isin(labels.keys())]
df = df.sample(frac=1, random_state=44).reset_index(drop=True) # shuffle
print(df.shape)

# create data directory
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=4)
folds = list(kf.split(df))
print('\n\n', folds, '\n\n')
for current_fold_i,current_fold in enumerate(folds): # for each fold
	for current_split,current_split_name in zip(current_fold, ['train', 'test']): # for each train/test split
		print(current_split_name, current_split)
		for i in current_split:
			row = df.iloc[i]
			if TEXT_COLUMN == 'title+abstract': # use combined title + abstract as input
				if row['title'] != row['title']: abstract = row['abstract'] # if NaN
				elif row['abstract'] != row['abstract']: abstract = row['title'] # if NaN
				else: abstract = row['title'] + ' ' + row['abstract']
			else: abstract = row[TEXT_COLUMN] # use abstract or title as input
			if abstract != abstract: continue # skip if no abstract (nan)
		
			# tokenize abstract
			abstract = tokenizer.tokenize(abstract)
			# remove stop words
			abstract = [token for token in abstract if token.lower() not in stoplist]
			abstract = ' '.join(abstract)
	
			# save as .txt file
			label = 'relevant' if labels[row['pubmed_id']] else 'irrelevant'
			command = 'echo "{}" > {}/{}/{}/{}/{}.txt'.format(abstract, OUTPUT_FOLDER, current_fold_i, current_split_name, label, i)
			os.system(command)

			if not i % 100: print('{} / {}'.format(i, df.shape[0]))












