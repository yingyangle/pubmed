#!/Users/christine/anaconda3/bin/python
# -*- coding: utf-8 -*-
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

import os, re, json, pandas as pd, numpy as np, warnings
import matplotlib.pyplot as plt
from palettable.wesanderson import FantasticFox1_5
from palettable.cartocolors.qualitative import Bold_10
import seaborn

seaborn.set()
warnings.filterwarnings('ignore')

METRIC = 'loss'
CHART_TITLE = 'PubMed Classification - BERT Models - '+METRIC.capitalize()
OUTPUT_FILE = 'bert_{}.png'.format(METRIC)
DATA_FILE = 'PubMed_BERT_Models.csv'

def graph(group_data, group_labels, legend_labels, legend_title='', title='grouped_bar'):
	# set plot dimensions, margin, and font
	plt.rcParams['figure.figsize'] = [16,9]
	plt.subplots_adjust(left=0.2, right=4, top=1, bottom=0.2)
	# format data
	df = pd.DataFrame(
		[[label]+data for label,data in zip(group_labels, group_data)],
		columns=[legend_title] + legend_labels
	)

	# plot grouped bar chart
	colors = Bold_10.hex_colors + [FantasticFox1_5.hex_colors[0]]
	ax = df.plot(kind='bar',
			stacked=False,
			width=0.8, # make bigger to decrease space between bars
			x=legend_title,
			color=colors,
			ylim=(0,100))
	x_offset = -0.06
	y_offset = -4
	for p in ax.patches:
		b = p.get_bbox()
		val = "{:.1f}".format(b.y1 + b.y0)
		ax.annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset), fontsize=16, color='white')

	plt.xticks(rotation=0, fontsize=20)
	plt.yticks(fontsize=20)
	plt.xlabel('',  fontsize=20)
	plt.ylabel(METRIC, fontsize=20)
	plt.title(title, fontsize=30)
	plt.legend(loc='upper left', bbox_to_anchor=(1,1), fontsize=16)
	plt.tight_layout()
	# plt.savefig('graphs/'+re.sub(' ', '_', title.lower())+'.png', dpi=200)
	plt.savefig(OUTPUT_FILE, dpi=200)
	# plt.show()
	return

df = pd.read_csv(DATA_FILE)
df = df.sort_values(by=['bert_model'])
df[METRIC] = df[METRIC]*100
data1 = list(df[df['dataset'] == '1'][METRIC])
data2 = list(df[df['dataset'] == '2'][METRIC])
data12 = list(df[df['dataset'] == '1+2'][METRIC])

group_data = [data1, data2, data12]
group_labels = ['Prev. Annotations', 'New Annotations', 'All Annotations']
legend_labels = sorted(list(set(df['bert_model'])))
print(group_data)
print(legend_labels)
graph(group_data, group_labels, legend_labels, 'Model', CHART_TITLE)




