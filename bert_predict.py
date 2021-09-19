#!/Users/christine/anaconda3/bin/python
# -*- coding: utf-8 -*-

# bert_predict.py
# loads saved fine-tuned model and gets predictions for data in bertdata/bertdata_*
# saves predictions as results/predictions_*.csv and results/predictions_*_raw.csv


###### Imports ######

import os, sys, shutil, json, pandas as pd
from datetime import date, datetime

import keras
import tensorflow as tf
import tensorflow_text as text

import matplotlib.pyplot as plt
import seaborn

seaborn.set()
tf.get_logger().setLevel('ERROR')
start = datetime.now()


##### SELECT DATASET #####

INPUT_TYPE = sys.argv[2]
TRAIN_DATASET = sys.argv[3]
UNANNOTATED_DATASET = sys.argv[4]
TRAIN_DATASET_FOLDER = 'bertdata/bertdata{TRAIN_DATASET}_{INPUT_TYPE}'
UNANNOTATED_DATASET_FOLDER = 'bertdata/bertdata{UNANNOTATED_DATASET}'
PUBMED_FOLDER = '.' # direct this to main pubmed folder

print('\npubmed folder:', PUBMED_FOLDER)
print('dataset:', TRAIN_DATASET)
print('unannotated dataset:', UNANNOTATED_DATASET)
print('dataset folder:', TRAIN_DATASET_FOLDER)
print('unannotated dataset folder:', UNANNOTATED_DATASET_FOLDER, '\n')

##### SELECT BERT MODEL #####

BERT_MODEL_SELECTED = sys.argv[1]

bert_models_dict = {
	'bert': 'bert_en_uncased_L-12_H-768_A-12',
	'smallbert': 'small_bert/bert_en_uncased_L-4_H-512_A-8',
	'albert': 'albert_en_base',
	'experts_pubmed': 'experts_pubmed',
	'electra': 'electra_base',
	'talkingheads': 'talking-heads_base',
}
BERT_MODEL_NAME = bert_models_dict[BERT_MODEL_SELECTED]
BERT_MODEL_NICKNAME = '{BERT_MODEL_SELECTED}{TRAIN_DATASET}_{INPUT_TYPE}'

print('bert model:', BERT_MODEL_NAME)
print('bert model nickname:', BERT_MODEL_NICKNAME)
print('\n\n')


###### Load saved model ######

model = keras.models.load_model(f'{PUBMED_FOLDER}/saved_models/'+BERT_MODEL_NICKNAME)


###### Get predictions ######

# get test data
texts = []
filenames = os.listdir(f'{PUBMED_FOLDER}/bertdata/bertdata_{UNANNOTATED_DATASET}_{INPUT_TYPE}')
for f in filenames:
	with open(f'{PUBMED_FOLDER}/bertdata/bertdata_{UNANNOTATED_DATASET}_{INPUT_TYPE}/{f}') as ein:
		t = ein.read().strip()
		texts.append(t)

# get predictions
predictions = model.predict(texts)
predictions = predictions.tolist()
predictions_thresh = [1 if tf.sigmoid(x) >= 0.5 else 0 for x in predictions]

# save results
df = pd.DataFrame({
	'pubmed_id': [x[:-4] for x in filenames],
	'relevance': predictions,
})
df.to_csv(f'{PUBMED_FOLDER}/results/predictions_{UNANNOTATED_DATASET}_{BERT_MODEL_NICKNAME}_raw.csv', index=False, encoding='utf-8-sig')
df = pd.DataFrame({
	'pubmed_id': [x[:-4] for x in filenames],
	'relevance': predictions_thresh,
})
df.to_csv(f'{PUBMED_FOLDER}/results/predictions_{UNANNOTATED_DATASET}_{BERT_MODEL_NICKNAME}.csv', index=False, encoding='utf-8-sig')
print(df)

print('\nRUNTIME:', str(datetime.now() - start))







