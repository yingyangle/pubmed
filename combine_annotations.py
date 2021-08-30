#!/Users/christine/anaconda3/bin/python
# -*- coding: utf-8 -*-

# combine_annotations.py
# combine annotations*.csv files and combine article_info*.csv files

import os, re, json, pandas as pd

os.chdir('data')

def combine_annotations():
	FILES = ['annotations1.csv', 'annotations2.csv']
	OUTPUT_FILE = 'annotations1+2.csv'
	dfs = []
	for f in FILES:
		df = pd.read_csv(f)
		df = df[df['relevance'].isin([0,1,3])] # only keep annotated rows
		dfs.append(df)

	df = pd.concat(dfs).astype({'relevance': 'int32'}) # make sure relevance is int
	df = df[['relevance', 'pubmed_id', 'title']]
	df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

	print(df.columns)
	print('\ntotal', df.shape)
	print('relevant', df[df['relevance'] == 1].shape)
	print('irrelevant', df[df['relevance'] == 0].shape)
	print('unsure', df[df['relevance'] == 3].shape, '\n')

	print('\n')
	print(df.dtypes)

def combine_article_info():
	FILES = ['article_info1.csv', 'article_info2.csv']
	OUTPUT_FILE = 'article_info1+2.csv'
	dfs = []
	for f in FILES:
		df = pd.read_csv(f)
		dfs.append(df)

	df = pd.concat(dfs)
	df = df[['pubmed_id', 'title', 'abstract', 'authors']]
	df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

	print(df.columns)
	print('\ntotal', df.shape)
	print('\n')
	print(df.dtypes)

combine_annotations()
combine_article_info()






