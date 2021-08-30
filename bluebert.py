# -*- coding: utf-8 -*-

import sys

##### SELECT DATASET #####

DATASET_NAME = sys.argv[1]
INPUT_TYPE = sys.argv[2]
DATASET_FOLDER = 'bertdata/bertdata{}_{}'.format(DATASET_NAME, INPUT_TYPE)
PUBMED_FOLDER = '/data/yangael/pubmed/'

print('\npubmed folder:', PUBMED_FOLDER)
print('dataset:', DATASET_NAME)
print('dataset folder:', DATASET_FOLDER)

##### SELECT BERT MODEL #####

BERT_MODEL_SELECTED = 'bluebert'
BERT_MODEL_NICKNAME = '{}{}_{}'.format(BERT_MODEL_SELECTED, DATASET_NAME, INPUT_TYPE)

print('bert model nickname:', BERT_MODEL_NICKNAME)
print('\n\n')

"""## Imports"""

# # Install transformers library.
# !pip install -q git+https://github.com/huggingface/transformers.git
# # Install helper functions.
# !pip install -q git+https://github.com/gmihaila/ml_things.git

import os, io, json, pandas as pd
from datetime import date, datetime

import torch
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from ml_things import plot_dict, plot_confusion_matrix, fix_text
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
from transformers import (AutoConfig,
													AutoModelForSequenceClassification,
													AutoTokenizer, AdamW,
													get_linear_schedule_with_warmup,
													set_seed,
													)

import matplotlib.pyplot as plt
import seaborn

seaborn.set()
start = datettime.now()

"""## Parameters"""

# Set seed for reproducibility,
set_seed(123)

# Number of training epochs (authors recommend between 2 and 4)
epochs = 4

# Number of batches - depending on the max sequence length and GPU memory.
# For 512 sequence length batch of 10 works without cuda memory issues.
# For small sequence length can try batch of 32 or higher.
batches = 16
batch_size = batches

# Pad or truncate text sequences to a specific length
# if `None` it will use maximum sequence of word piece tokens allowed by model.
max_length = 512

# Look for gpu to use. Will use `cpu` by default if no gpu found.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Name of transformers model - will use already pretrained model.
# Path of transformer model - will load your own model from local disk.
# model_name_or_path = 'bert-base-cased'
model_name_or_path = 'bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12'

# Dicitonary of labels and their id - this will be used to convert.
# String labels to number ids.
labels_ids = {'irrelevant': 0, 'relevant': 1}

# How many labels are we using in training.
# This is used to decide size of classification head.
n_labels = len(labels_ids)

"""## Functions"""

class PubMedDataset(Dataset):
	def __init__(self, path, use_tokenizer, labels_ids, max_sequence_len=None):
		if not os.path.isdir(path): # Check if path exists.
			raise ValueError('Invalid `path` variable! Needs to be a directory')
		# Check max sequence length.
		max_sequence_len = use_tokenizer.max_len if max_sequence_len is None else max_sequence_len
		texts = []
		labels = []
		print('Reading partitions...')
		# loop through each label
		for label, label_id,	in tqdm(labels_ids.items()):
			sentiment_path = os.path.join(path, label)

			files_names = os.listdir(sentiment_path)#[:10] # Sample for debugging.
			print('Reading %s files...' % label)
			# Go through each file and read its content
			for file_name in tqdm(files_names):
				file_path = os.path.join(sentiment_path, file_name)

				content = io.open(file_path, mode='r', encoding='utf-8').read()
				content = fix_text(content) # Fix any unicode issues
				texts.append(content)
				labels.append(label_id)

		# Number of examples
		self.n_examples = len(labels)
		# Use tokenizer on texts. This can take a while.
		print('Using tokenizer on all texts. This can take a while...')
		self.inputs = use_tokenizer(texts, add_special_tokens=True, truncation=True, padding=True, return_tensors='pt',	max_length=max_sequence_len)
		# Get maximum sequence length
		self.sequence_len = self.inputs['input_ids'].shape[-1]
		print('Texts padded or truncated to %d length!' % self.sequence_len)
		# Add labels
		self.inputs.update({'labels':torch.tensor(labels)})
		print('Finished!\n')

		return

	def __len__(self):
		return self.n_examples

	def __getitem__(self, item):
		return {key: self.inputs[key][item] for key in self.inputs.keys()}

def train(dataloader, optimizer_, scheduler_, device_):
	global model

	# Tracking variables.
	predictions_labels = []
	true_labels = []
	# Total loss for this epoch.
	total_loss = 0

	# Put the model into training mode.
	model.train()

	# For each batch of training data...
	for batch in tqdm(dataloader, total=len(dataloader)):

		# Add original labels - use later for evaluation.
		true_labels += batch['labels'].numpy().flatten().tolist()
		
		# move batch to device
		batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}
		
		# Always clear any previously calculated gradients before performing a
		# backward pass.
		model.zero_grad()

		# Perform a forward pass (evaluate the model on this training batch).
		# This will return the loss (rather than the model output) because we
		# have provided the `labels`.
		# The documentation for this a bert model function is here:
		# https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
		outputs = model(**batch)

		# The call to `model` always returns a tuple, so we need to pull the
		# loss value out of the tuple along with the logits. We will use logits
		# later to calculate training accuracy.
		loss, logits = outputs[:2]

		# Accumulate the training loss over all of the batches so that we can
		# calculate the average loss at the end. `loss` is a Tensor containing a
		# single value; the `.item()` function just returns the Python value
		# from the tensor.
		total_loss += loss.item()

		# Perform a backward pass to calculate the gradients.
		loss.backward()

		# Clip the norm of the gradients to 1.0.
		# This is to help prevent the "exploding gradients" problem.
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

		# Update parameters and take a step using the computed gradient.
		# The optimizer dictates the "update rule"--how the parameters are
		# modified based on their gradients, the learning rate, etc.
		optimizer.step()

		# Update the learning rate.
		scheduler.step()

		# Move logits and labels to CPU
		logits = logits.detach().cpu().numpy()

		# Convert these logits to list of predicted labels values.
		predictions_labels += logits.argmax(axis=-1).flatten().tolist()

	# Calculate the average loss over the training data.
	avg_epoch_loss = total_loss / len(dataloader)
	
	# Return all true labels and prediction for future evaluations.
	return true_labels, predictions_labels, avg_epoch_loss

def validation(dataloader, device_):
	# Use global variable for model.
	global model

	# Tracking variables
	predictions_labels = []
	true_labels = []
	#total loss for this epoch.
	total_loss = 0

	# Put the model in evaluation mode--the dropout layers behave differently
	# during evaluation.
	model.eval()

	# Evaluate data for one epoch
	for batch in tqdm(dataloader, total=len(dataloader)):

		# add original labels
		true_labels += batch['labels'].numpy().flatten().tolist()

		# move batch to device
		batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}

		# Telling the model not to compute or store gradients, saving memory and
		# speeding up validation
		with torch.no_grad():				

				# Forward pass, calculate logit predictions.
				# This will return the logits rather than the loss because we have
				# not provided labels.
				# token_type_ids is the same as the "segment ids", which
				# differentiates sentence 1 and 2 in 2-sentence tasks.
				# The documentation for this `model` function is here:
				# https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
				outputs = model(**batch)

				# The call to `model` always returns a tuple, so we need to pull the
				# loss value out of the tuple along with the logits. We will use logits
				# later to to calculate training accuracy.
				loss, logits = outputs[:2]
				
				# Move logits and labels to CPU
				logits = logits.detach().cpu().numpy()

				# Accumulate the training loss over all of the batches so that we can
				# calculate the average loss at the end. `loss` is a Tensor containing a
				# single value; the `.item()` function just returns the Python value
				# from the tensor.
				total_loss += loss.item()
				
				# get predicitons to list
				predict_content = logits.argmax(axis=-1).flatten().tolist()

				# update list
				predictions_labels += predict_content

	# Calculate the average loss over the training data.
	avg_epoch_loss = total_loss / len(dataloader)

	# Return all true labels and prediciton for future evaluations.
	return true_labels, predictions_labels, avg_epoch_loss

"""## Load BERT Model

"""

# Get model configuration.
print('Loading configuraiton...')
model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
																					num_labels=n_labels)

# Get model's tokenizer.
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)

# Get the actual model.
print('Loading model...')
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
																													 config=model_config)

# Load model to defined device.
model.to(device)
print('Model loaded to `%s`'%device)

"""## Format Data"""

print('Dealing with Train...')
# Create pytorch dataset.
train_dataset = PubMedDataset(path=PUBMED_FOLDER+DATASET_FOLDER+'/train',
															 use_tokenizer=tokenizer,
															 labels_ids=labels_ids,
															 max_sequence_len=max_length)
print('Created `train_dataset` with %d examples!'%len(train_dataset))

# Move pytorch dataset into dataloader.
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print('Created `train_dataloader` with %d batches!'%len(train_dataloader))

print()

print('Dealing with ...')
# Create pytorch dataset.
valid_dataset =	PubMedDataset(path=PUBMED_FOLDER+DATASET_FOLDER+'/test',
															 use_tokenizer=tokenizer,
															 labels_ids=labels_ids,
															 max_sequence_len=max_length)
print('Created `valid_dataset` with %d examples!'%len(valid_dataset))

# Move pytorch dataset into dataloader.
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
print('Created `eval_dataloader` with %d batches!'%len(valid_dataloader))

"""## Train Model"""

# Note: AdamW is a class from the huggingface library (as opposed to pytorch)
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
									lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
									eps = 1e-8 # args.adam_epsilon	- default is 1e-8.
									)

# Total number of training steps is number of batches * number of epochs.
# `train_dataloader` contains batched data so `len(train_dataloader)` gives
# us the number of batches.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
																						num_warmup_steps = 0, # Default value in run_glue.py
																						num_training_steps = total_steps)

# Store the average loss after each epoch so we can plot them.
all_loss = {'train_loss':[], 'val_loss':[]}
all_acc = {'train_acc':[], 'val_acc':[]}

# Loop through each epoch.
print('Epoch')
for epoch in tqdm(range(epochs)):
	print()
	print('Training on batches...')
	# Perform one full pass over the training set.
	train_labels, train_predict, train_loss = train(train_dataloader, optimizer, scheduler, device)
	train_acc = accuracy_score(train_labels, train_predict)

	# Get prediction form model on validation data.
	print('Validation on batches...')
	valid_labels, valid_predict, val_loss = validation(valid_dataloader, device)
	val_acc = accuracy_score(valid_labels, valid_predict)

	# Print loss and accuracy values to see how training evolves.
	print("	train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"%(train_loss, val_loss, train_acc, val_acc))
	print()

	# Store the loss value for plotting the learning curve.
	all_loss['train_loss'].append(train_loss)
	all_loss['val_loss'].append(val_loss)
	all_acc['train_acc'].append(train_acc)
	all_acc['val_acc'].append(val_acc)

# Save training history
history_dict = {**all_loss, **all_acc}
print('\n##########################\n')
print('\n\nTRAINING HISTORY\n')
print(history_dict, '\n\n')
with open(PUBMED_FOLDER+f'results/{BERT_MODEL_NICKNAME}.json', 'w') as aus:
	json.dump(history_dict, aus)

# Save model
try:
	os.mkdir(PUBMED_FOLDER+'saved_models/'+BERT_MODEL_NICKNAME)
except:
	print('\nFAILED TO CREATE FOLDER\n')
try:
	torch.save(model.state_dict(), PUBMED_FOLDER+'saved_models/'+BERT_MODEL_NICKNAME)
except:
	print('\nFAILED TO SAVE MODEL\n')

# Plot training history
try:
	from palettable.wesanderson import IsleOfDogs3_4

	COLOR1 = IsleOfDogs3_4.hex_colors[3] # training
	COLOR2 = IsleOfDogs3_4.hex_colors[2] # validation

	# get data to plot
	acc = history_dict['train_acc']
	val_acc = history_dict['val_acc']
	loss = history_dict['train_loss']
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
except:
	print("\nFAILED TO PLOT TRAINING HISTORY\n")

"""## Evaluate"""

# Get prediction form model on validation data. This is where you should use
# your test data.
true_labels, predictions_labels, avg_epoch_loss = validation(valid_dataloader, device)

# save predictions
try:
	predictions = model.predict(test_ds)
	print('\n### PREDICTIONS ###\n')
	print(type(predictions))
	predictions = predictions.tolist()
	with open(PUBMED_FOLDER+'results/predictions_{}.json'.format(BERT_MODEL_NICKNAME), 'w') as aus:
		json.dump(predictions, aus)
except:
	print('\nFAILED TO GET PREDICTIONS\n')

# Create the evaluation report.
evaluation_report = classification_report(true_labels, predictions_labels, labels=list(labels_ids.values()), target_names=list(labels_ids.keys()))
# Show the evaluation report.
print(evaluation_report)

# save accuracy and loss to .csv
precision, recall, fscore, support = score(true_labels, predictions_labels, average='macro')
print('\n\nprecision, recall, fscore, support:')
print(precision, recall, fscore, support, '\n\n')
df = pd.read_csv(PUBMED_FOLDER+'PubMed_BERT_Models_NEW.csv')
df = df.append(pd.DataFrame({
		'id': [BERT_MODEL_NICKNAME],
		'bert_model': [BERT_MODEL_SELECTED],
		'input_type': [INPUT_TYPE],
		'dataset': [DATASET_NAME],
		'accuracy': [fscore],
		'loss': [avg_epoch_loss],
		'date': [datetime.now().strftime("%d/%m/%Y %H:%M:%S")],
})).reset_index(drop=True)
df.to_csv(PUBMED_FOLDER+'PubMed_BERT_Models_NEW.csv', index=False, encoding='utf-8-sig')
df

# Plot confusion matrix.
plot_confusion_matrix(y_true=true_labels, y_pred=predictions_labels,
											classes=list(labels_ids.keys()), normalize=True,
											magnify=3,
											);
plt.savefig(PUBMED_FOLDER+'results/confusion_matrix_{}.png'.format(BERT_MODEL_NICKNAME), dpi=300)

print('\nRUNTIME:', str(datetime.now() - start)])