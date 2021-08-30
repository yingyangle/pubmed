#!/Users/christine/anaconda3/bin/python
# -*- coding: utf-8 -*-

# combine_articles.py
# combines PubMed articles retrieved from different search terms and removes articles that are already annotated
# saves resulting unnannotated articles as to_annotate.csv
# combines data/pubmed_articles_*.csv as a single to_annotate.csv
# combines data/authors_*.json files into a single article_authors.json, and also create authors.json with all unique authors

import os, re, json, pandas as pd

EXISTING_ANNOTATIONS_FILE = 'annotations1+2.csv'

os.chdir('data')

# combine pubmed_articles_*.csvs into to_annotate.csv
def combine_articles():
	# read in results csvs
	files = [x for x in os.listdir() if x.startswith('pubmed_articles_')]
	print(files)
	dfs = []
	for f in files:
		df_temp = pd.read_csv(f)
		dfs.append(df_temp)
	df = pd.concat(dfs)
	df = df.drop_duplicates()
	print(df.shape)

	df_annotated = pd.read_csv(EXISTING_ANNOTATIONS_FILE)
	df_annotated = df_annotated.drop_duplicates()
	annotated_ids = list(df_annotated['pubmed_id'])

	# remove articles that are already annotated
	df = df[~df['pubmed_id'].isin(annotated_ids)]
	print(df.shape, '\n')

	# save to .csv
	df.to_csv('to_annotate.csv', index=False, encoding='utf-8-sig')

# combine authors_*.json into authors.json
def combine_authors():
	# read in author jsons
	files = [x for x in os.listdir() if x.startswith('authors_')]
	print(files)
	author_data = {}
	article_author_data = {}
	for f in files:
		with open(f, 'r') as ein:
			authors = json.load(ein)
			article_author_data = {**article_author_data, **authors}
			for article in authors:
				for author in authors[article]:
					fname = str(author['firstname'])
					lname = str(author['lastname'])
					fullname = ' '.join([fname, lname])
					author_data[fullname] = author
	print('authors', len(author_data))
	print('articles', len(article_author_data))
	
	# save results to .json
	with open('authors.json', 'w') as aus:
		json.dump(author_data, aus)
	with open('article_authors.json', 'w') as aus:
		json.dump(article_author_data, aus)

# combine_articles()
combine_authors()







