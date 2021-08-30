#!/Users/christine/anaconda3/bin/python
# -*- coding: utf-8 -*-

# split_annotations_info.py
# split annotations file and article info file

import os, re, json, pandas as pd

FULL_FILE = 'to_annotate.csv'
ANNOTATIONS_FILE = 'annotations3.csv'
INFO_FILE = 'article_info3.csv'

os.chdir('data')

df = pd.read_csv(FULL_FILE)
df_annotations = df[['relevance', 'pubmed_id', 'title']]
df_info = df[['pubmed_id', 'publication_date', 'title', 'abstract', 'authors']]

df_annotations.to_csv(ANNOTATIONS_FILE, index=False, encoding='utf-8-sig')
df_info.to_csv(INFO_FILE, index=False, encoding='utf-8-sig')

