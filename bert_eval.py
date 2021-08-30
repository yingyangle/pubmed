#!/Users/christine/anaconda3/bin/python
# -*- coding: utf-8 -*-

# bert_eval.py
# loads saved fine-tuned model and tests it on data from bertdata/bertdata_*
# updates PubMed_BERT_Models_Eval.csv with evaluation results
# saves predictions as results/predictions_*_test*.json and results/predictions_*_test*_raw.json


###### Imports ######

import os, sys, shutil, json, pandas as pd
from datetime import date, datetime

import keras
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization # to create AdamW optimizer
from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt
import seaborn

seaborn.set()
tf.get_logger().setLevel('ERROR')


###### Define function ######

def evaluate_bert(BERT_MODEL_SELECTED, INPUT_TYPE, DATASET_NAME_TRAIN, DATASET_NAME_TEST):

	##### SELECT DATASET #####
	
	DATASET_FOLDER_TRAIN = 'bertdata/bertdata{}_{}'.format(DATASET_NAME_TRAIN, INPUT_TYPE)
	DATASET_FOLDER_TEST = 'bertdata/bertdata{}_{}'.format(DATASET_NAME_TEST, INPUT_TYPE)
	PUBMED_FOLDER = '/data/yangael/pubmed/'

	print('\npubmed folder:', PUBMED_FOLDER)
	print('train dataset:', DATASET_NAME_TRAIN)
	print('train dataset folder:', DATASET_FOLDER_TRAIN)
	print('test dataset:', DATASET_NAME_TEST)
	print('test dataset folder:', DATASET_FOLDER_TEST)

	##### SELECT BERT MODEL #####

	bert_models_dict = {
		'bert': 'bert_en_uncased_L-12_H-768_A-12',
		'smallbert': 'small_bert/bert_en_uncased_L-4_H-512_A-8',
		'albert': 'albert_en_base',
		'experts_pubmed': 'experts_pubmed',
		'electra': 'electra_base',
		'talkingheads': 'talking-heads_base',
	}
	BERT_MODEL_NAME = bert_models_dict[BERT_MODEL_SELECTED]
	BERT_MODEL_NICKNAME = '{}{}_{}'.format(BERT_MODEL_SELECTED, DATASET_NAME_TRAIN, INPUT_TYPE)

	print('bert model:', BERT_MODEL_NAME)
	print('bert model nickname:', BERT_MODEL_NICKNAME)
	print('\n\n')

	###### Format data ######

	AUTOTUNE = tf.data.AUTOTUNE
	batch_size = 32
	seed = 42
	
	# get training data
	raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
		PUBMED_FOLDER+DATASET_FOLDER_TRAIN+'/train',
		batch_size=batch_size,
		validation_split=0.2,
		subset='training',
		seed=seed)
	class_names = raw_train_ds.class_names
	train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

	# get validation data
	val_ds = tf.keras.preprocessing.text_dataset_from_directory(
		PUBMED_FOLDER+DATASET_FOLDER_TRAIN+'/train',
		batch_size=batch_size,
		validation_split=0.2,
		subset='validation',
		seed=seed)
	val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

	# get test data
	test_ds = tf.keras.preprocessing.text_dataset_from_directory(
		PUBMED_FOLDER+DATASET_FOLDER_TEST+'/test',
		batch_size=batch_size)
	test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
	

	###### Load saved model ######
	
	model = keras.models.load_model(PUBMED_FOLDER+'saved_models/'+BERT_MODEL_NICKNAME)
	
	###### Compile model ######
	
	### Loss function ###
	# use binary crossentropy for binary classification
	loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
	metrics = tf.metrics.BinaryAccuracy()

	### Optimizer ###
	epochs = 5
	steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
	num_train_steps = steps_per_epoch * epochs
	num_warmup_steps = int(0.1 * num_train_steps) # linear warm-up phase for first 10% of training steps
	init_lr = 3e-5 # small initial learning rate for fine-tuning (e.g. 5e-5, 3e-5, 2e-5)
	optimizer_type = 'adamw' # AdamW (Adaptive Moments) - uses weight decay

	optimizer = optimization.create_optimizer(init_lr=init_lr,
												num_train_steps=num_train_steps,
												num_warmup_steps=num_warmup_steps,
												optimizer_type=optimizer_type)

	model.compile(loss=loss,
					optimizer=optimizer,
					metrics=metrics)
	print(model.summary())
	

	###### Evaluate model ######

	### evaluate model on test data using evaluate()
	loss, accuracy = model.evaluate(test_ds)
	print(f'\n\nAccuracy: {accuracy}')
	print(f'Loss: {loss}\n')

	### evaluate model on test data using predict()
	# format data for predict()
	texts = []
	labels = []
	for x in test_ds:
		texts.extend(list(x[0]))
		labels.extend(x[1])
	texts = [str(x) for x in texts]
	labels = [int(x) for x in labels]

	# get predictions
	predictions = model.predict(texts)
	predictions = predictions.tolist()
	predictions_thresh = [1 if tf.sigmoid(x) >= 0.5 else 0 for x in predictions]

	# interpret and save predictions
	with open(f'{PUBMED_FOLDER}results/predictions_{BERT_MODEL_NICKNAME}_test{DATASET_NAME_TEST}_raw.json', 'w') as aus:
		json.dump(predictions, aus)
	with open(f'{PUBMED_FOLDER}results/predictions_{BERT_MODEL_NICKNAME}_test{DATASET_NAME_TEST}.json', 'w') as aus:
		json.dump(predictions_thresh, aus)

	# evaluate predictions
	precision, recall, fscore, support = precision_recall_fscore_support(labels, predictions_thresh, average='macro')
	print(f'\n\nPrecision: {accuracy}')
	print(f'Recall: {recall}')
	print(f'F Score: {fscore}')
	print(f'Support: {support}\n')

	# save accuracy and loss to .csv
	df = pd.read_csv(PUBMED_FOLDER+'PubMed_BERT_Models_Eval.csv')
	df = df.append(pd.DataFrame({
		'id': [BERT_MODEL_NICKNAME],
		'bert_model': [BERT_MODEL_SELECTED],
		'input_type': [INPUT_TYPE],
		'dataset_train': [DATASET_NAME_TRAIN],
		'dataset_test': [DATASET_NAME_TEST],
		'precision': [precision],
		'recall': [recall],
		'fscore': [fscore],
		'accuracy': [accuracy],
		'loss': [loss],
		'date': [datetime.now().strftime("%d/%m/%Y %H:%M:%S")]
	})).reset_index(drop=True)
	df.to_csv(PUBMED_FOLDER+'PubMed_BERT_Models_Eval.csv', index=False, encoding='utf-8-sig')
	print(df, '\n')



###### Execute ######

models = [
	'bert',
	'smallbert',
	'albert',
	'experts_pubmed',
	'electra',
	'talkingheads',
]

input_types = [
	'title',
	'abstract',
	'title+abstract',
]

# run evaluate_bert() for all models using test data from the dataset as the train data (no overlap between train and test data though)
def run_all():
	for BERT_MODEL_SELECTED in models:
		for INPUT_TYPE in input_types:
			for DATASET_NAME in ['1', '2', '3']:
				print('\n\n**********************************')
				print(BERT_MODEL_SELECTED, INPUT_TYPE, DATASET_NAME)
				print('**********************************\n')
				evaluate_bert(BERT_MODEL_SELECTED, INPUT_TYPE, DATASET_NAME, DATASET_NAME)
				# try:
				# 	print('\n\n**********************************')
				# 	print(BERT_MODEL_SELECTED, INPUT_TYPE, DATASET_NAME)
				# 	print('**********************************\n')
				# 	evaluate_bert(BERT_MODEL_SELECTED, INPUT_TYPE, DATASET_NAME, DATASET_NAME)
				# except:
				# 	print('\n####################################')
				# 	print('FAILED:', BERT_MODEL_SELECTED, INPUT_TYPE, DATASET_NAME)
				# 	print('####################################\n\n')

# run evaluate_bert() for all models using test data from a different dataset from the train data
def run_all_different_datasets():
	for BERT_MODEL_SELECTED in models:
		for INPUT_TYPE in input_types:
			for TRAIN_DATASET in ['1', '2']:
				if TRAIN_DATASET == '1': TEST_DATASET = '2'
				else: TEST_DATASET = '1'
				try:
					print('\n\n**********************************')
					print(BERT_MODEL_SELECTED, INPUT_TYPE, TRAIN_DATASET)
					print('**********************************\n')
					evaluate_bert(BERT_MODEL_SELECTED, INPUT_TYPE, TRAIN_DATASET, TEST_DATASET)
				except:
					print('\n####################################')
					print('FAILED:', BERT_MODEL_SELECTED, INPUT_TYPE, TRAIN_DATASET)
					print('####################################\n\n')


# run evaluate_bert() on all models
if len(sys.argv) == 1: run_all_different_datasets()
elif sys.argv[1] == 'run_all': run_all()

# run evaluate_bert() only once for selected parameters
elif len(sys.argv) > 2:
	BERT_MODEL_SELECTED = sys.argv[1]
	INPUT_TYPE = sys.argv[2]
	TRAIN_DATASET = sys.argv[3]
	TEST_DATASET = sys.argv[4]
	evaluate_bert(BERT_MODEL_SELECTED, INPUT_TYPE, TRAIN_DATASET, TEST_DATASET)



	







