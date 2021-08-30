# -*- coding: utf-8 -*-

import os, re, sys

files = [x for x in os.listdir() if 'pbs.e' in x]
print(files)

job_to_delete = []
for e_file in files:
	with open(e_file, 'r') as ein:
		text = ein.read()
		if 'Terminated\n' in text:
			o_file = re.sub('pbs.e', 'pbs.o', e_file)
			os.system('rm '+e_file)
			os.system('rm '+o_file)
			print(e_file, o_file)
			
			job = f[f.find('pbs.e')+5:]
			job_to_delete.append(job)

print('jobs deleted:')
print(job_to_delete)
