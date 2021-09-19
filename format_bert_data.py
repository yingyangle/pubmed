#!/Users/christine/anaconda3/bin/python
# -*- coding: utf-8 -*-

import os, sys, shutil, re, json, pandas as pd
from datetime import datetime
from sklearn.model_selection import KFold
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

start = datetime.now()
tokenizer = RegexpTokenizer(r'\w+')

# customizing the nltk English stop list
stoplist = stopwords.words('english')
stoplist.extend(["ever", "one", "do","does","make", "go", "us", "to", "get", "about", "may", "s", ".", ",", "!", "i", "I", '\"', "?", ";", "--", "--", "would", "could", "”", "Mr.", "Miss", "Mrs.", "don’t", "said", "can't", "didn't", "aren't", "I'm", "you're", "they're", "'s"])

# format dataset for 80/20 train test split
def format_dataset_split(INPUT_TYPE, DATASET_NAME):
	ANNOTATIONS_FILE = f'data/annotations{DATASET_NAME}.csv'
	ARTICLE_INFO = f'data/article_info{DATASET_NAME}.csv'
	OUTPUT_FOLDER = f'bertdata/bertdata{DATASET_NAME}_{INPUT_TYPE}'
	TRAIN_TEST_SPLIT = 0.8

	print('input type:', INPUT_TYPE)
	print('dataset:', DATASET_NAME)
	print('annotations file:', ANNOTATIONS_FILE)
	print('article info:', ARTICLE_INFO)
	print('output folder:', OUTPUT_FOLDER)

	# create output folders
	try: # delete folder if already there
		shutil.rmtree(OUTPUT_FOLDER)
		shutil.rmtree(OUTPUT_FOLDER+'/test')
	except: pass
	if not os.path.isdir(OUTPUT_FOLDER): os.mkdir(OUTPUT_FOLDER)
	os.mkdir(OUTPUT_FOLDER+'/train')
	os.mkdir(OUTPUT_FOLDER+'/test')
	os.mkdir(OUTPUT_FOLDER+'/train/relevant')
	os.mkdir(OUTPUT_FOLDER+'/train/irrelevant')
	os.mkdir(OUTPUT_FOLDER+'/test/relevant')
	os.mkdir(OUTPUT_FOLDER+'/test/irrelevant')

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
	CURRENT_FOLDER = 'train'
	for i,row in df.iterrows():
		if i == THRESHOLD: CURRENT_FOLDER = 'test'
	
		pid = int(row['pubmed_id'])
		if pid not in labels.keys(): continue
	
		if INPUT_TYPE == 'title+abstract': # use combined title + abstract as input
			if row['title'] != row['title']: abstract = row['abstract'] # if NaN
			elif row['abstract'] != row['abstract']: abstract = row['title'] # if NaN
			else: abstract = row['title'] + ' ' + row['abstract']
		else: abstract = row[INPUT_TYPE] # use abstract or title as input
		if abstract != abstract: continue # skip if no abstract (nan)
		# tokenize abstract
		abstract = tokenizer.tokenize(abstract)
		# remove stop words
		abstract = [token for token in abstract if token.lower() not in stoplist]
		abstract = ' '.join(abstract)
	
		# save as .txt file
		label = 'relevant' if labels[pid] else 'irrelevant'
		command = f'echo {abstract} > {OUTPUT_FOLDER}/{CURRENT_FOLDER}/{label}/{i}.txt'
		os.system(command)
	
		if not i % 100: print(f'{i} / {df.shape[0]}')

# format dataset for cross validation
def format_dataset_CV(INPUT_TYPE, DATASET_NAME, NUM_FOLDS):
	ANNOTATIONS_FILE = f'data/annotations{DATASET_NAME}.csv'
	ARTICLE_INFO = f'data/article_info{DATASET_NAME}.csv'
	OUTPUT_FOLDER = f'bertdata/bertdata{DATASET_NAME}_{INPUT_TYPE}_CV'

	print('input type:', INPUT_TYPE)
	print('dataset:', DATASET_NAME)
	print('annotations file:', ANNOTATIONS_FILE)
	print('article info:', ARTICLE_INFO)
	print('output folder:', OUTPUT_FOLDER)

	# create output folders
	try: shutil.rmtree(OUTPUT_FOLDER) # delete folder if already there
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
	
	# read in labels
	df = pd.read_csv(ANNOTATIONS_FILE)
	df = df[df['relevance'] < 2]
	labels = {int(row['pubmed_id']):int(row['relevance']) for i,row in df.iterrows()}

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
				if INPUT_TYPE == 'title+abstract': # use combined title + abstract as input
					if row['title'] != row['title']: abstract = row['abstract'] # if NaN
					elif row['abstract'] != row['abstract']: abstract = row['title'] # if NaN
					else: abstract = row['title'] + ' ' + row['abstract']
				else: abstract = row[INPUT_TYPE] # use abstract or title as input
				if abstract != abstract: continue # skip if no abstract (nan)
		
				# tokenize abstract
				abstract = tokenizer.tokenize(abstract)
				# remove stop words
				abstract = [token for token in abstract if token.lower() not in stoplist]
				abstract = ' '.join(abstract)
	
				# save as .txt file
				label = 'relevant' if labels[row['pubmed_id']] else 'irrelevant'
				command = f'echo "{abstract}" > {OUTPUT_FOLDER}/{current_fold_i}/{current_split_name}/{label}/{i}.txt'
				os.system(command)

				if not i % 100: print(f'{i} / {df.shape[0]}')

# format dataset for no data split
def format_dataset_full(INPUT_TYPE, DATASET_NAME):
	ANNOTATIONS_FILE = f'data/annotations{DATASET_NAME}.csv'
	ARTICLE_INFO = f'data/article_info{DATASET_NAME}.csv'
	OUTPUT_FOLDER_BASE = f'bertdata/bertdata{DATASET_NAME}_{INPUT_TYPE}'
	OUTPUT_FOLDER = f'bertdata/bertdata{DATASET_NAME}_{INPUT_TYPE}/full'

	print('input type:', INPUT_TYPE)
	print('dataset:', DATASET_NAME)
	print('annotations file:', ANNOTATIONS_FILE)
	print('article info:', ARTICLE_INFO)
	print('output folder:', OUTPUT_FOLDER)

	# create folders
	try: shutil.rmtree(OUTPUT_FOLDER) # delete folder if already there
	except: pass
	if not os.path.isdir(OUTPUT_FOLDER_BASE): os.mkdir(OUTPUT_FOLDER_BASE)
	os.mkdir(OUTPUT_FOLDER)
	os.mkdir(OUTPUT_FOLDER+'/relevant')
	os.mkdir(OUTPUT_FOLDER+'/irrelevant')

	# read in labels
	df = pd.read_csv(ANNOTATIONS_FILE)
	df = df[df['relevance'] < 2]
	labels = {int(row['pubmed_id']):int(row['relevance']) for i,row in df.iterrows()}

	# read in abstracts and titles
	df = pd.read_csv(ARTICLE_INFO)
	df = df.dropna(subset=['pubmed_id'])
	df['pubmed_id'] = df['pubmed_id'].astype('int')
	df = df[df['pubmed_id'].isin(labels.keys())]
	df = df.sample(frac=1, random_state=44).reset_index(drop=True) # shuffle
	print(df.shape)

	# create data directory
	for i,row in df.iterrows():
		pid = int(row['pubmed_id'])
		if pid not in labels.keys(): continue
	
		if INPUT_TYPE == 'title+abstract': # use combined title + abstract as input
			if row['title'] != row['title']: abstract = row['abstract'] # if NaN
			elif row['abstract'] != row['abstract']: abstract = row['title'] # if NaN
			else: abstract = row['title'] + ' ' + row['abstract']
		else: abstract = row[INPUT_TYPE] # use abstract or title as input
		if abstract != abstract: continue # skip if no abstract (nan)
		# tokenize abstract
		abstract = tokenizer.tokenize(abstract)
		# remove stop words
		abstract = [token for token in abstract if token.lower() not in stoplist]
		abstract = ' '.join(abstract)
	
		# save as .txt file
		label = 'relevant' if labels[pid] else 'irrelevant'
		command = f'echo {abstract} > {OUTPUT_FOLDER}/{label}/{i}.txt'
		os.system(command)
	
		if not i % 100: print(f'{i} / {df.shape[0]}')


# format dataset for no data split and no labels
def format_dataset_unannotated(INPUT_TYPE, NICKNAME):
	OUTPUT_FOLDER = f'bertdata/bertdata_{NICKNAME}_{INPUT_TYPE}'

	print('\ninput type:', INPUT_TYPE)
	print('output folder:', OUTPUT_FOLDER, '\n')

	# create output folder
	try: os.system('rm -rf '+OUTPUT_FOLDER) # delete folder if already there
	except: pass
	os.mkdir(OUTPUT_FOLDER)

	tokenizer = RegexpTokenizer(r'\w+')

	# customizing the nltk English stop list
	stoplist = stopwords.words('english')
	stoplist.extend(["ever", "one", "do","does","make", "go", "us", "to", "get", "about", "may", "s", ".", ",", "!", "i", "I", '\"', "?", ";", "--", "--", "would", "could", "”", "Mr.", "Miss", "Mrs.", "don’t", "said", "can't", "didn't", "aren't", "I'm", "you're", "they're", "'s"])

	# read in abstracts and titles
	df = pd.read_csv(ARTICLE_INFO)
	df = df.dropna(subset=['pubmed_id'])
	df['pubmed_id'] = df['pubmed_id'].astype('int')
	print(df.shape)

	# create data directory
	for i,row in df.iterrows():
		if INPUT_TYPE == 'title+abstract': # use combined title + abstract as input
			if row['title'] != row['title']: abstract = row['abstract'] # if NaN
			elif row['abstract'] != row['abstract']: abstract = row['title'] # if NaN
			else: abstract = row['title'] + ' ' + row['abstract']
		else: abstract = row[INPUT_TYPE] # use abstract or title as input
		if abstract != abstract: continue # skip if no abstract (nan)
		# tokenize abstract
		abstract = tokenizer.tokenize(abstract)
		# remove stop words
		abstract = [token for token in abstract if token.lower() not in stoplist]
		abstract = ' '.join(abstract)
	
		# save as .txt file
		os.system('echo {} > {}/{}.txt'.format(abstract, OUTPUT_FOLDER, row['pubmed_id']))
	
		if not i % 100: print('{} / {}'.format(i, df.shape[0]))



### execute ###

datasets = [
	'1',
	'2',
	'1+2',
]

input_types = [
	'title',
	'abstract',
	'title+abstract',
]

if len(sys.argv) <= 1:
	print('*** Please use one of the following string arguments when running the script, in order to select which action to run:')
	print('\tsplit\n\tcv\n\tfull\n\tunannotated\n')
else:
	if sys.argv[1] == 'split':
		for DATASET_NAME in datasets:
			for INPUT_TYPE in input_types: format_dataset_split(INPUT_TYPE, DATASET_NAME)
	elif sys.argv[1] == 'cv':
		for DATASET_NAME in datasets:
			for INPUT_TYPE in input_types: format_dataset_CV(INPUT_TYPE, DATASET_NAME, 5)
	elif sys.argv[1] == 'full':
		for DATASET_NAME in datasets:
			for INPUT_TYPE in input_types: format_dataset_full(INPUT_TYPE, DATASET_NAME)
	elif sys.argv[1] == 'unannotated':
		NICKNAME = input('what do you want to name this unannotated dataset? ')
		for INPUT_TYPE in input_types: format_dataset_unannotated(INPUT_TYPE, NICKNAME)

print('\nRUNTIME:', str(datetime.now() - start))





