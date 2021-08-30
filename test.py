import os, shutil, json, pandas as pd
from datetime import date, datetime

import keras
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization # to create AdamW optimizer
from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt
import seaborn
from palettable.wesanderson import IsleOfDogs3_4
