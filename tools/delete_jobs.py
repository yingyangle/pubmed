#!/Users/christine/anaconda3/bin/python
# -*- coding: utf-8 -*-

import os, sys

start = int(sys.argv[1])
end = int(sys.argv[2])

for job_id in range(start, end+1):
	command = f'qdel {job_id}'
	print(command)
	os.system(command)

