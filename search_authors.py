#!/Users/christine/anaconda3/bin/python
# -*- coding: utf-8 -*-

# search_authors.py
# search for more articles by relevant authors

import os, re, sys, json, pandas as pd, time
from datetime import date, datetime
from pymed import PubMed
pubmed = PubMed(tool="PubMedSearcher", email="cyang1999@gmail.com")

start = datetime.now()
print('start', start)

OUTPUT_FOLDER = 'data/relevant_authors_articles'

# read in authors to search
with open(f'{OUTPUT_FOLDER}/authors_relevant.json', 'r') as ein:
	authors_to_search = json.load(ein)
print('\n# total relevant authors:', len(authors_to_search))

# read in authors already searched
try:
	with open(f'{OUTPUT_FOLDER}/authors_already_searched.json', 'r') as ein:
		authors_already_searched = json.load(ein)[:-3] # re-search last 3 authors
	print('# authors already searched:', len(authors_already_searched))
except: authors_already_searched = []

# read in articles already saved
try:
	with open(f'{OUTPUT_FOLDER}/articles_already_saved.json', 'r') as ein:
		articles_already_saved = json.load(ein)
	print('# articles already saved:', len(articles_already_saved))
except: articles_already_saved = []

# read in failed authors
try:
	with open(f'{OUTPUT_FOLDER}/failed_authors.json', 'r') as ein:
		failed_authors = json.load(ein)
	print('# failed authors so far:', len(failed_authors))
except: failed_authors = []

# remove authors already searched
authors_to_search = {k:v for k,v in authors_to_search.items() if k not in authors_already_searched}
print('# authors to search:', len(authors_to_search))
print('\n#####################\n')


# save articles retrieved for a single author
def save_results(author_fullname, author_articles):
	global authors_already_searched, articles_already_saved, failed_authors
	author_filename = re.sub(' ', '_', author_fullname)
	
	# save results to .json and .csv
	with open(f'{OUTPUT_FOLDER}/saved_articles/json/{author_filename}.json', 'w') as aus:
		json.dump(author_articles, aus, indent=2)

	df = pd.DataFrame({
		'fullname': [x['fullname'] for x in author_articles],
		'firstname': [x['firstname'] for x in author_articles],
		'lastname': [x['lastname'] for x in author_articles],
		'initials': [x['initials'] for x in author_articles],
		'affiliation': [x['affiliation'] for x in author_articles],
		'pubmed_id': [x['pubmed_id'] for x in author_articles],
		'publication_date': [x['publication_date'] for x in author_articles],
		'title': [x['title'] for x in author_articles],
		'abstract': [x['abstract'] for x in author_articles],
		'authors': [x['authors'] for x in author_articles],
		'doi': [x['doi'] for x in author_articles],
	})
	df.to_csv(f'{OUTPUT_FOLDER}/saved_articles/csv/{author_filename}.csv', index = None, encoding="utf-8")
	
	# update list of authors already searched
	with open(f'{OUTPUT_FOLDER}/authors_already_searched.json', 'w') as aus:
		json.dump(authors_already_searched, aus)
	
	# update list of articles already saved
	with open(f'{OUTPUT_FOLDER}/articles_already_saved.json', 'w') as aus:
		json.dump(articles_already_saved, aus)
	
	# update list of failed authors
	with open(f'{OUTPUT_FOLDER}/failed_authors.json', 'w') as aus:
		json.dump(failed_authors, aus)
	

# search articles by each author of this current article
i = 0
# for each author
for author_fullname in authors_to_search:
	author = authors_to_search[author_fullname]
	author_articles = []
	try:
		# get search term
		fname = author['firstname']
		lname = author['lastname']
		SEARCH_TERM = f'{lname}, {fname}[Author]'
		MAX_RESULTS = 100000

		# check if author already searched
		if author_fullname in authors_already_searched: 
			print('***** SKIPPED', author_fullname)
			continue
	
		# search for articles by this author
		results = pubmed.query(SEARCH_TERM, max_results=MAX_RESULTS)
	
		# format each article result
		for result in results:
			article = result.toDict()
			try: publication_date = article['publication_date'].strftime("%Y-%m-%d")
			except: publication_date = '-1'
			author_articles.append({
				'fullname': author_fullname,
				'firstname': author['firstname'],
				'lastname': author['lastname'],
				'initials': author['initials'],
				'affiliation': author['affiliation'],
				'pubmed_id': int(article['pubmed_id'].partition('\n')[0]),
				'publication_date': publication_date,
				'title': article['title'],
				'abstract': article['abstract'],
				'authors': article['authors'],
				'doi': article['doi'],
			})
			articles_already_saved.append(article['doi'])
	except:
		print('\n\n######### FAILED #########\n')
		print(author)
		print(f'\n######### total failed: {len(failed_authors)} #########\n\n')
		failed_authors.append(author)
	
	# save results
	authors_already_searched.append(author_fullname)
	save_results(author_fullname, author_articles)
	
	# save progress
	i += 1
	if not i % 100:
		print(f'{len(articles_already_saved)} articles saved, {len(authors_already_searched)} / {len(authors_to_search)} authors searched')
	
	time.sleep(1)

print('\nRUNTIME:', str(datetime.now() - start))











