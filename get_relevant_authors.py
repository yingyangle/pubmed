#!/Users/christine/anaconda3/bin/python
# -*- coding: utf-8 -*-

# get_relevant_authors.py
# get authors for articles marked as relevant in annotations and predictions, given a predictions file
# saves results as data/authors_relevant.json

import os, re, sys, json, pandas as pd, time
from datetime import datetime
from statistics import mean

PREDICTED_ARTICLES = sys.argv[1]
predictions = pd.read_csv(PREDICTED_ARTICLES)
predictions = predictions[predictions['relevance'] == 1]
pubmed_ids = [str(x) for x in predictions['pubmed_id']]
print(predictions.shape)
print(predictions.columns)

annotations = pd.read_csv('data/annotations1+2.csv')
annotations = annotations[annotations['relevance'] == 1]
pubmed_ids.extend([str(x) for x in annotations['pubmed_id']])
print(annotations.shape)
print(annotations.columns)

print('\npubmed_ids:', len(pubmed_ids))

with open('data/article_authors.json', 'r') as ein:
	article_authors_dict = json.load(ein)

num_articles = 0
authors_per_article = []
author_data = {}
for pubmed_id in pubmed_ids:
	if pubmed_id not in article_authors_dict: continue
	authors = article_authors_dict[pubmed_id]
	# save info for each author
	for author in authors:
		fname = str(author['firstname'])
		lname = str(author['lastname'])
		fullname = ' '.join([fname, lname])
		author_data[fullname] = author
	# increment stats
	authors_per_article.append(len(authors))
	num_articles += 1

print('\n\n# articles:', num_articles)
print('# authors:', len(author_data))
print('avg. # authors per article:', mean(authors_per_article))

# save as .json
with open('data/authors_relevant_new.json', 'w') as aus:
	json.dump(author_data, aus)

print('\nRUNTIME:', str(datetime.now() - start))








