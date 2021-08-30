#!/Users/christine/anaconda3/bin/python
# -*- coding: utf-8 -*-

# bert_CV.py
# does same thing as bert.py but uses cross validation
# fine-tunes a given BERT model to classify the relevance of PubMed articles
# takes in formatted train and test data from bertdata/bertdata_*
# updates PubMed_BERT_Models_CV.csv with evaluation results
# saves training history as results/training_history_*.json and results/training_history_*.png graph
# saves predictions as results/predictions_*.json and results/predictions_*_raw.json
# saves fine-tuned model in /saved_models

# electra1_title+abstract - pred
# electra2_abstract
# talkingheads1+2_title+abstract - pred
# bert2_title+abstract
# bert1_title+abstract

import sys

##### SELECT DATASET #####

INPUT_TYPE = sys.argv[2]
DATASET_NAME = sys.argv[3]
NUM_FOLDS = int(sys.argv[4]) # num folds for cross val, must match bertdata
DATASET_FOLDER = 'bertdata/bertdata{}_{}_CV'.format(DATASET_NAME, INPUT_TYPE)
PUBMED_FOLDER = '/data/yangael/pubmed/'

print('\npubmed folder:', PUBMED_FOLDER)
print('dataset:', DATASET_NAME)
print('dataset folder:', DATASET_FOLDER)
print('CV:', NUM_FOLDS, 'folds\n\n')

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
BERT_MODEL_NICKNAME_BASE = '{}{}_{}'.format(BERT_MODEL_SELECTED, DATASET_NAME, INPUT_TYPE)

print('bert model:', BERT_MODEL_NAME)
print('bert model nickname:', BERT_MODEL_NICKNAME_BASE)
print('\n\n')


###### Imports ######

import os, shutil, json, pandas as pd
from datetime import date, datetime
from statistics import mean

import keras
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization # to create AdamW optimizer
from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt
import seaborn
from palettable.wesanderson import IsleOfDogs3_4

seaborn.set()
tf.get_logger().setLevel('ERROR')
start = datetime.date()


###### Define function ######

# format data, train bert model, and evaluate bert model for one CV fold
def run_fold(CURRENT_FOLD):
	BERT_MODEL_NICKNAME = f'{BERT_MODEL_NICKNAME_BASE}_CV{CURRENT_FOLD}'
	fold_start = datetime.date()
	
	###### Format data ######

	AUTOTUNE = tf.data.AUTOTUNE
	batch_size = 32
	seed = 42

	# get training data
	raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
		f'{PUBMED_FOLDER}{DATASET_FOLDER}/{CURRENT_FOLD}/train',
		batch_size=batch_size,
		validation_split=0.2,
		subset='training',
		seed=seed)
	class_names = raw_train_ds.class_names
	train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

	# get validation data
	val_ds = tf.keras.preprocessing.text_dataset_from_directory(
		f'{PUBMED_FOLDER}{DATASET_FOLDER}/{CURRENT_FOLD}/train',
		batch_size=batch_size,
		validation_split=0.2,
		subset='validation',
		seed=seed)
	val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

	# get test data
	test_ds = tf.keras.preprocessing.text_dataset_from_directory(
		f'{PUBMED_FOLDER}{DATASET_FOLDER}/{CURRENT_FOLD}/test',
		batch_size=batch_size)
	test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

	# example texts
	for text_batch, label_batch in train_ds.take(1):
		for i in range(3):
			print(f'\nAbstract: {text_batch.numpy()[i]}')
			label = label_batch.numpy()[i]
			print(f'Relevance : {label} ({class_names[label]})')


	###### Load BERT model ######
	# https://tfhub.dev/google/collections/experts/bert/1

	map_name_to_handle = {
		'bert_en_uncased_L-12_H-768_A-12':
			'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
		'bert_en_cased_L-12_H-768_A-12':
			'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
		'bert_multi_cased_L-12_H-768_A-12':
			'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
		'small_bert/bert_en_uncased_L-2_H-128_A-2':
			'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
		'small_bert/bert_en_uncased_L-2_H-256_A-4':
			'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
		'small_bert/bert_en_uncased_L-2_H-512_A-8':
			'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
		'small_bert/bert_en_uncased_L-2_H-768_A-12':
			'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
		'small_bert/bert_en_uncased_L-4_H-128_A-2':
			'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
		'small_bert/bert_en_uncased_L-4_H-256_A-4':
			'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
		'small_bert/bert_en_uncased_L-4_H-512_A-8':
			'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
		'small_bert/bert_en_uncased_L-4_H-768_A-12':
			'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
		'small_bert/bert_en_uncased_L-6_H-128_A-2':
			'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
		'small_bert/bert_en_uncased_L-6_H-256_A-4':
			'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
		'small_bert/bert_en_uncased_L-6_H-512_A-8':
			'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
		'small_bert/bert_en_uncased_L-6_H-768_A-12':
			'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
		'small_bert/bert_en_uncased_L-8_H-128_A-2':
			'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
		'small_bert/bert_en_uncased_L-8_H-256_A-4':
			'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
		'small_bert/bert_en_uncased_L-8_H-512_A-8':
			'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
		'small_bert/bert_en_uncased_L-8_H-768_A-12':
			'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
		'small_bert/bert_en_uncased_L-10_H-128_A-2':
			'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
		'small_bert/bert_en_uncased_L-10_H-256_A-4':
			'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
		'small_bert/bert_en_uncased_L-10_H-512_A-8':
			'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
		'small_bert/bert_en_uncased_L-10_H-768_A-12':
			'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
		'small_bert/bert_en_uncased_L-12_H-128_A-2':
			'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
		'small_bert/bert_en_uncased_L-12_H-256_A-4':
			'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
		'small_bert/bert_en_uncased_L-12_H-512_A-8':
			'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
		'small_bert/bert_en_uncased_L-12_H-768_A-12':
			'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
		'albert_en_base':
			'https://tfhub.dev/tensorflow/albert_en_base/2',
		'electra_small':
			'https://tfhub.dev/google/electra_small/2',
		'electra_base':
			'https://tfhub.dev/google/electra_base/2',
		'experts_pubmed':
			'https://tfhub.dev/google/experts/bert/pubmed/2',
		'experts_wiki_books':
			'https://tfhub.dev/google/experts/bert/wiki_books/2',
		'talking-heads_base':
			'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
	}

	map_model_to_preprocess = {
		'bert_en_uncased_L-12_H-768_A-12':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'bert_en_cased_L-12_H-768_A-12':
			'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
		'small_bert/bert_en_uncased_L-2_H-128_A-2':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'small_bert/bert_en_uncased_L-2_H-256_A-4':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'small_bert/bert_en_uncased_L-2_H-512_A-8':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'small_bert/bert_en_uncased_L-2_H-768_A-12':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'small_bert/bert_en_uncased_L-4_H-128_A-2':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'small_bert/bert_en_uncased_L-4_H-256_A-4':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'small_bert/bert_en_uncased_L-4_H-512_A-8':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'small_bert/bert_en_uncased_L-4_H-768_A-12':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'small_bert/bert_en_uncased_L-6_H-128_A-2':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'small_bert/bert_en_uncased_L-6_H-256_A-4':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'small_bert/bert_en_uncased_L-6_H-512_A-8':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'small_bert/bert_en_uncased_L-6_H-768_A-12':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'small_bert/bert_en_uncased_L-8_H-128_A-2':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'small_bert/bert_en_uncased_L-8_H-256_A-4':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'small_bert/bert_en_uncased_L-8_H-512_A-8':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'small_bert/bert_en_uncased_L-8_H-768_A-12':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'small_bert/bert_en_uncased_L-10_H-128_A-2':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'small_bert/bert_en_uncased_L-10_H-256_A-4':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'small_bert/bert_en_uncased_L-10_H-512_A-8':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'small_bert/bert_en_uncased_L-10_H-768_A-12':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'small_bert/bert_en_uncased_L-12_H-128_A-2':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'small_bert/bert_en_uncased_L-12_H-256_A-4':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'small_bert/bert_en_uncased_L-12_H-512_A-8':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'small_bert/bert_en_uncased_L-12_H-768_A-12':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'bert_multi_cased_L-12_H-768_A-12':
			'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
		'albert_en_base':
			'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
		'electra_small':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'electra_base':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'experts_pubmed':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'experts_wiki_books':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		'talking-heads_base':
			'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	}

	tfhub_handle_encoder = map_name_to_handle[BERT_MODEL_NAME]
	tfhub_handle_preprocess = map_model_to_preprocess[BERT_MODEL_NAME]

	print(f'\n\nBERT model selected:		 {tfhub_handle_encoder}')
	print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')


	### Preprocessing model ###
	# Load preprocessing model to prepare text inputs for BERT. Transforms text inputs into numeric token ids and arranges them in tensors.

	# load BERT preprocessing model
	bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)

	# example output
	# input truncated to 128 tokens by preprocessor (see shape)
	text_test = ['there are many disparities in epidemiology']
	text_preprocessed = bert_preprocess_model(text_test)

	print(f'\nKeys	   : {list(text_preprocessed.keys())}')
	print(f'Shape	  : {text_preprocessed["input_word_ids"].shape}')
	print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
	print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
	print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}') # single sentence input, so only one value


	### Encoding model ###

	# load BERT encoding model
	bert_model = hub.KerasLayer(tfhub_handle_encoder)

	# example output
	bert_results = bert_model(text_preprocessed)
	print(f'\nLoaded BERT: {tfhub_handle_encoder}')

	# sentence embedding for entire abstract - use this for fine-tuning
	print(f'\nPooled Outputs Shape:{bert_results["pooled_output"].shape}') 
	print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')

	# contextual embedding for each token
	print(f'\nSequence Outputs Shape:{bert_results["sequence_output"].shape}')
	print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')

	# immediate activations of the L transformer blocks
	# each is a tensor of shape [batch_size, seq_length, 1024]
	# last value in list = sequence output
	print(f'\nSequence Outputs Values:{bert_results["encoder_outputs"][0].shape}')
	print(f'Sequence Outputs Values:{bert_results["encoder_outputs"][0]}')


	###### Define model ######

	def build_model():
		text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
		preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
		encoder_inputs = preprocessing_layer(text_input)
		encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
		outputs = encoder(encoder_inputs)
		net = outputs['pooled_output']
		net = tf.keras.layers.Dropout(0.1)(net)
		net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
		return tf.keras.Model(text_input, net)

	model = build_model()
	bert_raw_result = model(tf.constant(text_test))
	print(tf.sigmoid(bert_raw_result))

	# # show model structure
	# # tf.keras.utils.plot_model(model)
	# tf.keras.utils.plot_model(model, to_file=PUBMED_FOLDER+'bert_layers.png', dpi=200)

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


	###### Train model ######

	# train model
	print(f'\n\nTraining model with {tfhub_handle_encoder}\n')
	history = model.fit(x=train_ds,
						validation_data=val_ds,
						epochs=epochs)

	# save fine-tuned model for future use
	model.save(f'{PUBMED_FOLDER}saved_models/{BERT_MODEL_NICKNAME}')


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
	print('predictions length check', len(texts), len(labels), len(predictions), len(predictions_thresh))

	# interpret and save predictions
	with open(f'{PUBMED_FOLDER}results/predictions_{BERT_MODEL_NICKNAME}_raw.json', 'w') as aus:
		json.dump(predictions, aus)
	with open(f'{PUBMED_FOLDER}results/predictions_{BERT_MODEL_NICKNAME}.json', 'w') as aus:
		json.dump(predictions_thresh, aus)

	# evaluate predictions
	precision, recall, fscore, support = precision_recall_fscore_support(labels, predictions_thresh, average='macro')
	print(f'\n\nPrecision: {precision}')
	print(f'Recall: {recall}')
	print(f'F Score: {fscore}')
	print(f'Support: {support}\n')

	# save accuracy and loss to .csv
	df = pd.read_csv(PUBMED_FOLDER+'PubMed_BERT_Models_CV.csv')
	df = df.append(pd.DataFrame({
		'id': [BERT_MODEL_NICKNAME],
		'model_id': [BERT_MODEL_NICKNAME_BASE],
		'fold': [CURRENT_FOLD],
		'bert_model': [BERT_MODEL_SELECTED],
		'input_type': [INPUT_TYPE],
		'dataset': [DATASET_NAME],
		'precision': [precision],
		'recall': [recall],
		'fscore': [fscore],
		'accuracy': [accuracy],
		'loss': [loss],
		'date': [datetime.now().strftime("%d/%m/%Y %H:%M:%S")],
		'time_taken': [str(datetime.now() - fold_start)],
	})).reset_index(drop=True)
	df.to_csv(PUBMED_FOLDER+'PubMed_BERT_Models_CV.csv', index=False, encoding='utf-8-sig')
	print(df, '\n')


	###### Plot training history ######

	# save training history to .json
	history_dict = history.history
	with open(PUBMED_FOLDER+'results/training_history_{}.json'.format(BERT_MODEL_NICKNAME), 'w') as aus:
		json.dump(history_dict, aus)

	COLOR1 = IsleOfDogs3_4.hex_colors[3] # training
	COLOR2 = IsleOfDogs3_4.hex_colors[2] # validation

	# get data to plot
	acc = history_dict['binary_accuracy']
	val_acc = history_dict['val_binary_accuracy']
	loss = history_dict['loss']
	val_loss = history_dict['val_loss']

	epochs = range(1, len(acc) + 1)

	# format plot
	fig = plt.figure(figsize=(12, 9))
	plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.08)
	fig.subplots_adjust(hspace=.3)
	fig.tight_layout()

	# loss plot
	plt.subplot(2, 1, 1)
	plt.plot(epochs, loss, COLOR1, label='Training loss')
	plt.plot(epochs, val_loss, COLOR2, label='Validation loss')
	plt.title('Training and validation loss', fontsize=20)
	plt.ylabel('Loss', fontsize=20)
	plt.legend(fontsize=14)

	# accuracy plot
	plt.subplot(2, 1, 2)
	plt.plot(epochs, acc, COLOR1, label='Training acc')
	plt.plot(epochs, val_acc, COLOR2, label='Validation acc')
	plt.title('Training and validation accuracy', fontsize=20)
	plt.xlabel('Epochs', fontsize=20)
	plt.ylabel('Accuracy', fontsize=20)
	plt.legend(loc='lower right', fontsize=14)

	# save image
	fig.savefig(PUBMED_FOLDER+'results/training_history_{}.png'.format(BERT_MODEL_NICKNAME), dpi=300)


	###### Load & test saved model ######

	# # load saved model'
	# reloaded_model = keras.models.load_model(PUBMED_FOLDER+'saved_models/'+BERT_MODEL_NICKNAME)
	#
	# # test some example sentences on the model
	# def test_sentences(inputs, results):
	# 	result_for_printing = \
	# 	[f'input: {inputs[i]:<30} : score: {results[i][0]:.6f}'
	# 		for i in range(len(inputs))]
	# 	print(*result_for_printing, sep='\n')
	#
	# # sentences to try
	# examples = [
	# 	'there are many disparities in epidemiology',
	# ]
	#
	# # compare results from reloaded model and model in local memory
	# reloaded_results = tf.sigmoid(reloaded_model(tf.constant(examples)))
	# original_results = tf.sigmoid(model(tf.constant(examples)))
	#
	# print('\n\nResults from saved model:')
	# test_sentences(examples, reloaded_results)
	# print('\nResults from model in local memory:')
	# test_sentences(examples, original_results)


###### Execute ######

# execute for each CV fold
for CURRENT_FOLD in range(NUM_FOLDS):
	print('\n\n####################################')
	print('Fold', CURRENT_FOLD)
	print('####################################\n\n')
	run_fold(CURRENT_FOLD)
	
	# get avg of results from each fold
	df = pd.read_csv(PUBMED_FOLDER+'PubMed_BERT_Models_CV.csv')
	df = df[df['model_id']==BERT_MODEL_NICKNAME_BASE]
	print(df.shape)
	print(df)
	df_avg = pd.read_csv(PUBMED_FOLDER+'PubMed_BERT_Models_CV_avg.csv')
	df_avg = df_avg.append(pd.DataFrame({
		'id': [BERT_MODEL_NICKNAME_BASE],
		'bert_model': [BERT_MODEL_SELECTED],
		'input_type': [INPUT_TYPE],
		'dataset': [DATASET_NAME],
		'precision': [mean(df['precision'])],
		'recall': [mean(df['recall'])],
		'fscore': [mean(df['fscore'])],
		'accuracy': [mean(df['accuracy'])],
		'loss': [mean(df['loss'])],
		'date': [datetime.now().strftime("%d/%m/%Y %H:%M:%S")],
		'time_taken': [str(datetime.now() - start)],
	})).reset_index(drop=True)
	df_avg.to_csv(PUBMED_FOLDER+'PubMed_BERT_Models_CV_avg.csv', index=False, encoding='utf-8-sig')
	print(df_avg, '\n')

print('\nRUNTIME:', str(datetime.now() - start)])






