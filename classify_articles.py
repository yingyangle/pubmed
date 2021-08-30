#!/Users/christine/anaconda3/bin/python
# -*- coding: utf-8 -*-

import os, re, json, numpy as np, pandas as pd, warnings
from datetime import datetime
import gensim
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from palettable.wesanderson import FantasticFox1_5
from palettable.cartocolors.qualitative import Bold_10
import seaborn

seaborn.set()
warnings.filterwarnings('ignore')
start = datetime.now()

BATCH = '2'
CHART_TITLE = 'PubMed Classification (w2v)'
FILENAME = 'w2v_classification{}.png'.format(BATCH)
ANNOTATIONS_FILE = 'data/annotations{}.csv'.format(BATCH)
ARTICLE_INFO = 'data/article_info{}.csv'.format(BATCH)
print('batch:', BATCH)
print('filename:', FILENAME)
print('annotations:', ANNOTATIONS_FILE)
print('article info:', ARTICLE_INFO)
print('chart title:', CHART_TITLE, '\n\n')

# read in annotated labels (relevance)
df = pd.read_csv(ANNOTATIONS_FILE)
df = df[df['relevance'] < 2]
labels = {str(row['pubmed_id']):int(row['relevance']) for i,row in df.iterrows()}

## Here I am just customizing the nltk English stop list
stoplist = stopwords.words('english')
stoplist.extend(["ever", "one", "do","does","make", "go", "us", "to", "get", "about", "may", "s", ".", ",", "!", "i", "I", '\"', "?", ";", "--", "--", "would", "could", "”", "Mr.", "Miss", "Mrs.", "don’t", "said", "can't", "didn't", "aren't", "I'm", "you're", "they're", "'s"])

# read in article info (abstracts and titles)
df = pd.read_csv(ARTICLE_INFO)
df = df.dropna(subset=['abstract']) # drop if abstract is NaN
df = df[df['pubmed_id'].isin(labels.keys())]
abstracts = list(df['abstract'])
pubmed_ids = list(df['pubmed_id'])
titles = list(df['title'])
relevance = [labels[str(x)] if str(x) in labels else 10 for x in df['pubmed_id']]

wordlist = {}

# Split the documents into tokens.
tokenizer = RegexpTokenizer(r'\w+')
for idx in range(len(abstracts)):
	abstracts[idx] = tokenizer.tokenize(abstracts[idx])  # Split into words.
	for w in abstracts[idx]:
		if w not in wordlist.keys():
			wordlist[w] = 1

# remove stop words
abstracts = [[token for token in abstract if token.lower() not in stoplist] for abstract in abstracts]

# load w2v model
bigmodel = gensim.models.KeyedVectors.load_word2vec_format("PubMed-and-PMC-w2v.bin", binary=True)

# and just get the words we need, save to a dictionary
smallmodel = {}
for w in wordlist:
	if w in bigmodel:
		smallmodel[w] = bigmodel[w]

abstractvectors = []   # this list will contain one 300-dimensional vector per headline

for h in abstracts:
	totvec = np.zeros(200)
	for w in h:
		if w in bigmodel:
			totvec = totvec + smallmodel[w]
	abstractvectors.append(totvec)



####### CLASSIFICATION #######


# execute
X_train, X_test, y_train, y_test = train_test_split(abstractvectors, relevance, test_size=0.8)
X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

print(sorted(list(set(y_train))))

models = [
	{
		'model': LogisticRegression(solver='liblinear', multi_class='ovr'),
		'name': 'Logistic Regression',
	},
	{
		'model': KNeighborsClassifier(3),
		'name': 'K Neighbors (k=3)',
	},
	{
		'model': SVC(kernel="linear", C=0.025),
		'name': 'Linear SVM',
	},
	{
		'model': SVC(gamma=2, C=1),
		'name': 'RBF SVM',
	},
	{
		'model': GaussianNB(),
		'name': 'Gaussian NB',
	},
	{
		'model': GaussianProcessClassifier(1.0 * RBF(1.0)),
		'name': 'Gaussian Process',
	},
	{
		'model': DecisionTreeClassifier(max_depth=5),
		'name': 'Decision Tree',
	},
	{
		'model': RandomForestClassifier(n_estimators = 400, max_features = 3, oob_score = True),
		'name': 'Random Forest',
	},
	{
		'model': AdaBoostClassifier(),
		'name': 'Ada Boost',
	},
	{
		'model': MLPClassifier(batch_size=8, learning_rate="adaptive", solver="sgd", max_iter=100, hidden_layer_sizes=200),
		'name': 'MLP Neural Net',
	},
	{
		'model': QuadraticDiscriminantAnalysis(),
		'name': 'QDA',
	},
]

precisions = []
recalls = []
fscores = []

# train and test each model
for m in models:
	model = m['model']
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	scores_results = classification_report(y_test, y_pred)
	precision, recall, fscore, support = score(y_test, y_pred, average='macro')
	print(m['name'])
	print(scores_results)
	# print('precision', precision)
	# print('recall', recall)
	# print('fscore', fscore, '\n\n')
	precisions.append(precision)
	recalls.append(recall)
	fscores.append(fscore)

# graph model results
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
	df.plot(kind='bar',
			stacked=False,
			width=0.8, # make bigger to decrease space between bars
			x=legend_title,
			color=colors,
			ylim=(0,1))
	plt.xticks(rotation=0, fontsize=20)
	plt.xlabel('')
	plt.ylabel('Percentage', fontsize=20)
	plt.title(title, fontsize=30)
	plt.legend(loc='upper left', bbox_to_anchor=(1,1))
	plt.tight_layout()
	# plt.savefig('graphs/'+re.sub(' ', '_', title.lower())+'.png', dpi=200)
	plt.savefig(FILENAME, dpi=200)
	# plt.show()
	return

group_data = [precisions, recalls, fscores]
group_labels = ['Precision', 'Recall', 'F Score']
legend_labels = [x['name'] for x in models]
graph(group_data, group_labels, legend_labels, 'Model', CHART_TITLE)

print('\nRUNTIME:', str(datetime.now() - start)])


