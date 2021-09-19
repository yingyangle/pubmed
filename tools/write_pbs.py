#!/Users/christine/anaconda3/bin/python
# -*- coding: utf-8 -*-

import os, sys, re

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

datasets = [
	'1',
	'2',
	'1+2'
]

datasets2 = [
	'1',
	'2'
]

# read in template .pbs file
with open('tools/template.pbs', 'r') as ein:
	template = ein.read() + '\n'

# try all combinations for bert.py
def run_bert(write_pbs=True):
	for BERT_MODEL_SELECTED in models:
		for INPUT_TYPE in input_types:
			for DATASET in datasets:
				command = f"python bert.py '{BERT_MODEL_SELECTED}' '{INPUT_TYPE}' '{DATASET}'"
				job_name = f'train_{BERT_MODEL_SELECTED}{DATASET}_{INPUT_TYPE}.pbs'
				# write .pbs file
				if write_pbs:
					pbs = template + command + '\n'
					with open('tools/pbs/'+job_name, 'w') as aus:
						aus.write(pbs)
				# push to queue
				os.system('qsub tools/pbs/'+job_name)
				print(job_name)

# try all combinations for bertCV.py
def run_bert_CV(write_pbs=True, CV=5):
	for BERT_MODEL_SELECTED in models:
		for INPUT_TYPE in input_types:
			for DATASET in datasets:
				command = f"python bert_CV.py '{BERT_MODEL_SELECTED}' '{INPUT_TYPE}' '{DATASET}' '{CV}'"
				job_name = f'CV_{BERT_MODEL_SELECTED}{DATASET}_{INPUT_TYPE}.pbs'
				# write .pbs file
				if write_pbs:
					pbs = template + command + '\n'
					with open('tools/pbs/'+job_name, 'w') as aus:
						aus.write(pbs)
				# push to queue
				os.system('qsub tools/pbs/'+job_name)
				print(job_name)

# try all combinations for ml_models.py
def run_ml(write_pbs=True):
	for INPUT_TYPE in input_types:
		for DATASET in datasets:
			command = f"python ml.py '{INPUT_TYPE}' '{DATASET}'"
			job_name = f'ml{DATASET}_{INPUT_TYPE}.pbs'
			# write .pbs file
			if write_pbs:
				pbs = template + command + '\n'
				with open('tools/pbs/'+job_name, 'w') as aus:
					aus.write(pbs)
			# push to queue
			os.system('qsub tools/pbs/'+job_name)
			print(job_name)

if len(sys.argv) <= 1:
	print('*** Please use one of the following string arguments when running the script, in order to select which action to run:')
	print('\tbert_split\n\tbert_cv\n\tml_split\n')
else:
	# if sys.argv[1] == 'run_create_datasets': run_create_datasets()
	if sys.argv[1] == 'bert_split': run_bert()
	elif sys.argv[1] == 'bert_cv': run_bert_CV()
	elif sys.argv[1] == 'ml_split': run_ml()











