#!/Users/christine/anaconda3/bin/python
# -*- coding: utf-8 -*-

# create_data_dir_test.py
# create data directory for testing BERT model on unannotated data
# save formatted data folder as bertdata/bertdata_*

import os, sys, pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

os.chdir('data')

TEXT_COLUMN = sys.argv[1] # use 'abstract' or 'title' or 'title+abstract' as training data
ARTICLE_INFO = sys.argv[2] # .csv file containing article info (e.g. to_annotate.csv)
NICKNAME = sys.argv[3] # nickname for this dataset
OUTPUT_FOLDER = f'../bertdata/bertdata_{NICKNAME}_{TEXT_COLUMN}'

print('\ninput type:', TEXT_COLUMN)
print('article info:', ARTICLE_INFO)
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
	os.system('echo {} > {}/{}.txt'.format(abstract, OUTPUT_FOLDER, row['pubmed_id']))
	
	if not i % 100: print('{} / {}'.format(i, df.shape[0]))




