#!/Users/christine/anaconda3/bin/python
# -*- coding: utf-8 -*-

import os, re, sys, json, pandas as pd
from pymed import PubMed
pubmed = PubMed(tool="PubMedSearcher", email="yangael@bc.edu")

## PUT YOUR SEARCH TERM HERE ##
SEARCH_TERM = sys.argv[1]
MAX_RESULTS = 100000000 #100000
results = pubmed.query(SEARCH_TERM, max_results=MAX_RESULTS)
articleList = []
articleInfo = []
authorInfo = {}

print(SEARCH_TERM, MAX_RESULTS)

counter = 0

for article in results:

# We need to convert article to dictionary.
	articleDict = article.toDict()
	articleList.append(articleDict)

# Generate list of dict records which will hold all article details

for article in articleList:

# Sometimes article['pubmed_id'] contains a list separated with comma
# Take the first pubmedId in that list.

	pubmedId = article['pubmed_id'].partition('\n')[0]
	if counter % 100 == 0:
		print("articles retrieved: ", counter)
	counter += 1

	# Append article info to dictionary
	articleInfo.append({u'pubmed_id':pubmedId,
						u'publication_date':article['publication_date'],
						u'title':article['title'],
						u'abstract':article['abstract'],
						u'authors':article['authors']})
	authorInfo[pubmedId] = article['authors']

# Generate Pandas DataFrame from list of dictionaries
articlesPD = pd.DataFrame.from_dict(articleInfo)

# save results
export_csv = articlesPD.to_csv (r'pubmed_articles_{}_{}.csv'.format(SEARCH_TERM, MAX_RESULTS), index = None, header=True, encoding="utf-8")
with open(r'authors_{}_{}.json'.format(SEARCH_TERM, MAX_RESULTS), 'w') as aus:
	json.dump(authorInfo, aus)

# Print first 10 rows of dataframe
# print(articlesPD.head(10))


