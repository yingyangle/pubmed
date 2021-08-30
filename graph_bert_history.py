#!/Users/christine/anaconda3/bin/python
# -*- coding: utf-8 -*-

import os, re, json, numpy as np, pandas as pd, warnings
import matplotlib.pyplot as plt, seaborn
from palettable.wesanderson import IsleOfDogs3_4
seaborn.set()

PUBMED_FOLDER = ''
bert_model_name_short = 'smallbert'

with open('training_history_smallbert.json', 'r') as ein:
    history_dict = json.load(ein)

COLOR1 = IsleOfDogs3_4.hex_colors[3]
COLOR2 = IsleOfDogs3_4.hex_colors[2]

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
fig.savefig(PUBMED_FOLDER+'training_history_{}.png'.format(bert_model_name_short), dpi=300)

