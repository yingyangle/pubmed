#!/Users/christine/anaconda3/bin/python
# -*- coding: utf-8 -*-

# search_authors.py
# search for more articles by relevant authors

import os, re, sys, json, pandas as pd, time
from datetime import date, datetime
from pymed import PubMed
pubmed = PubMed(tool="PubMedSearcher", email="cyang1999@gmail.com")

os.chdir('data')

# read in authors to search
with open('authors_relevant.json', 'r') as ein:
	authors_to_search = json.load(ein)
print('\n# relevant authors:', len(authors_to_search))

# read in authors and articles already searched
with open('articles_by_relevant_authors.json', 'r') as ein:
	article_info = json.load(ein)
	authors_already_searched = [str(x['firstname'])+' '+str(x['lastname']) for x in article_info]
	authors_already_searched = authors_already_searched[:-2] # research last 2 authors
print('# authors already searched:', len(authors_already_searched))
print('# articles already saved:', len(article_info), '\n')

# save current articles retrieved 
def save_results():
	# save results to .json and .csv
	with open('articles_by_relevant_authors.json', 'w') as aus:
		json.dump(article_info, aus)

	df = pd.DataFrame({
		'firstname': [x['firstname'] for x in article_info],
		'lastname': [x['lastname'] for x in article_info],
		'initials': [x['initials'] for x in article_info],
		'affiliation': [x['affiliation'] for x in article_info],
		'pubmed_id': [x['pubmed_id'] for x in article_info],
		'publication_date': [x['publication_date'] for x in article_info],
		'title': [x['title'] for x in article_info],
		'abstract': [x['abstract'] for x in article_info],
		'authors': [x['authors'] for x in article_info],
		'doi': [x['doi'] for x in article_info],
	})
	df.to_csv('articles_by_relevant_authors.csv', index = None, encoding="utf-8")

# search articles by each author of this current article
i = 0
for fullname in authors_to_search:
	author = authors_to_search[fullname]
	try:
		fname = author['firstname']
		lname = author['lastname']
		SEARCH_TERM = f'{lname}, {fname}[Author]'
		MAX_RESULTS = 100000
		# print(SEARCH_TERM)

		if fname+' '+lname in authors_already_searched: 
			print('***** SKIPPED', fname, lname)
			continue
		results = pubmed.query(SEARCH_TERM, max_results=MAX_RESULTS)
		for result in results:
			article = result.toDict()
			try: publication_date = article['publication_date'].strftime("%Y-%m-%d")
			except: publication_date = '-1'
			article_info.append({
				'fullname': fullname,
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
			# print(article)
	except:
		print('\n\n######### FAILED #########\n')
		print(author)
		print('\n#########\n\n')
	
	# save progress
	if not i % 500: 
		save_results()
		print(len(article_info), 'articles saved')
	
	time.sleep(3)
		


print('finished')










