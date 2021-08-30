#!/Users/christine/anaconda3/bin/python
# -*- coding: utf-8 -*-

import os, re

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
with open('template.pbs', 'r') as ein:
	template = ein.read() + '\n'

# try all combinations for bert.py
def run_bert(write_pbs=True):
	for BERT_MODEL_SELECTED in models:
		for INPUT_TYPE in input_types:
			for DATASET in datasets:
				command = f"python3 bert.py '{BERT_MODEL_SELECTED}' '{INPUT_TYPE}' '{DATASET}'"
				job_name = f'train_{BERT_MODEL_SELECTED}{DATASET}_{INPUT_TYPE}.pbs'
				# write .pbs file
				if write_pbs:
					pbs = template + command + '\necho "FINISHED"\n'
					with open('pbs/'+job_name, 'w') as aus:
						aus.write(pbs)
				# push to queue
				os.system('qsub pbs/'+job_name)
				print(job_name)

# try all combinations for bertCV.py
def run_bert_CV(write_pbs=True, CV=5):
	for BERT_MODEL_SELECTED in models:
		for INPUT_TYPE in input_types:
			for DATASET in datasets:
				command = f"python3 bert_CV.py '{BERT_MODEL_SELECTED}' '{INPUT_TYPE}' '{DATASET}' '{CV}'"
				job_name = f'CV_{BERT_MODEL_SELECTED}{DATASET}_{INPUT_TYPE}.pbs'
				# write .pbs file
				if write_pbs:
					pbs = template + command + '\necho "FINISHED"\n'
					with open('pbs/'+job_name, 'w') as aus:
						aus.write(pbs)
				# push to queue
				os.system('qsub pbs/'+job_name)
				print(job_name)

def run_bluebert(write_pbs=True):
	for INPUT_TYPE in input_types:
		for DATASET in datasets:
			command = f"python3 bluebert.py '{DATASET}' '{INPUT_TYPE}'"
			job_name = f'bluebert_{DATASET}_{INPUT_TYPE}.pbs'
			# write .pbs file
			if write_pbs:
				pbs = template + command + '\necho "FINISHED"\n'
				with open('pbs/'+job_name, 'w') as aus:
					aus.write(pbs)
			# push to queue
			os.system('qsub pbs/'+job_name)
			print(job_name)

# try all combinations for eval_bert.py
def run_eval(write_pbs=True):
	for BERT_MODEL_SELECTED in models:
		for INPUT_TYPE in input_types:
			for TRAIN_DATASET in datasets2:
				for TEST_DATASET in datasets2:
					command = f"python3 bert_eval.py ''{BERT_MODEL_SELECTED}' '{INPUT_TYPE}' '{TRAIN_DATASET}' '{TEST_DATASET}'"
					job_name = f'eval_{BERT_MODEL_SELECTED}_{INPUT_TYPE}_train{TRAIN_DATASET}_test{TEST_DATASET}.pbs'
					# write .pbs file
					if write_pbs:
						pbs = template + command + '\necho "FINISHED"\n'
						with open('pbs/'+job_name, 'w') as aus:
							aus.write(pbs)
					# push to queue
					os.system('qsub pbs/'+job_name)
					print(job_name)

run_bert()
# run_bluebert(0)
# run_eval(0)
# run_bert_CV()









