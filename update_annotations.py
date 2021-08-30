#!/Users/christine/anaconda3/bin/python
# -*- coding: utf-8 -*-

# update_annotations.py
# get corrected annotations from other annotators and update existing annotation files

import os, re, pandas as pd

OLD_ANNOTATIONS = 'data/annotations2.csv'
NEW_ANNOTATIONS = 'etc/annotations2_NNA.csv'

df = pd.read_csv(OLD_ANNOTATIONS)
df_new = pd.read_csv(NEW_ANNOTATIONS)
df_new = df_new[['relevance', 'pubmed_id']]
print(df.shape)

df = df.merge(df_new, how='left', left_on='pubmed_id', right_on='pubmed_id', suffixes=('_old', '_new'))
df['relevance_old'] = [row['relevance_new'] if row['relevance_new'] == row['relevance_new'] else row['relevance_old'] for i,row in df.iterrows()]

df['relevance_old'] = df['relevance_old'].astype('int32')
df = df.rename(columns={'relevance_old':'relevance'})
df = df.drop(['relevance_new'], axis=1)

# save to .csv
df.to_csv(OLD_ANNOTATIONS[:-4]+'_udpated.csv', index=False, encoding='utf-8-sig')

print(df.shape)