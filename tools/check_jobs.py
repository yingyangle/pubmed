#!/Users/christine/anaconda3/bin/python
# -*- coding: utf-8 -*-

# check_jobs.py
# check that all jobs completed successfully, indicated by "RUNTIME:" line at the end of the output file

import os, sys, re

# output files to check
files = [x for x in os.listdir() if 'pbs.o' in x]

# check a single file
def check(f):
	with open(f, 'r') as ein:
		text = ein.read().strip()
	lines = text.split('\n')
	if 'RUNTIME:' not in lines[-1]: print(f, re.sub('pbs.o', 'pbs.e', f))

# execute
for f in files:
	check(f)




