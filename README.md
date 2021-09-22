# PubMed Health Disparity

** **(currently still collecting articles -- will send email update when finished!)** **

Health disparity is an important research area focusing on studying the health outcomes for people of disadvantaged identities and backgrounds. One issue however is that many researchers who study health disparities often move on to study other areas of health research due to a lack of NIH funding. Additionally, many of the researchers who study health disparities are members of those disadvantaged communities themselves, which could be a possible indicator for some sort of discrimination in the funding selection process.

In this project, we want to see if the data reflects the trend of researchers moving away from studying health disparities by analyzing the articles about health disparity on PubMed. Our role in this project is to procure the data needed for this analysis. One way to retrieve these articles is to take articles matching the search term "disparity" on PubMed, however this does not always return results relevant to health disparity, and may include articles about other kinds of disparity. In order to get around this, we manually annotate the "relevance" of each article and train a model to predict whether or not a given article is about health disparity. 

### Procedure 

1. Get PubMed articles matching the keywords "disparity", "inequity", and  "inequality" 
	> [Article retrieval](#article-retrieval)
2. Manually annotate 2,000 relevant articles and 2,000 irrelevant articles 
	> [Annotation](#annotation) and [Formatting data](#formatting-data)
3. Train a model on our annotated data to predict the relevance of a given article
	> [ML Classification Models](#ml-classification-models) and [BERT Classification Models](#bert-classification-models)
4. Retrieve all PubMed articles matching our three search terms
	> [Getting relevant articles](#getting-relevant-articles)
5. Run the model on these articles to predict their relevance 
	> [Getting relevant articles](#getting-relevant-articles)
6. Get the authors for the articles predicted to be relevant 
	> [Getting authors of relevant articles](#getting-authors-of-relevant-articles)
7. Search PubMed for all other articles by these authors
	> [Getting articles by relevant authors](#getting-articles-by-relevant-authors)
8. Run the model on all these articles to predict their relevance 
	> [Predicting relevance of articles by relevant authors](#predicting-relevance-of-articles-by-relevant-authors)


## Overview

- [Setup](#setup)
- [Datasets](#datasets)
	- [Article retrieval](#article-retrieval)
	- [Annotation](#annotation)
	- [Formatting data](#formatting-data)
- [ML Classification Models](#ml-classification-models)
- [BERT Classification Models](#bert-classification-models)
	- [Prepare data for training a model](#prepare-data-for-training-a-model)
	- [Train and evaluate a model](#train-and-evaluate-a-model)
	- [Evaluate a model on a different dataset](#evaluate-a-model-on-a-different-dataset)
- [Searching for Relevant Articles](#searching-for-relevant-articles)
	- [Getting relevant articles](#getting-relevant-articles)
	- [Getting authors of relevant articles](#getting-authors-of-relevant-articles)
	- [Getting articles by relevant authors](#getting-articles-by-relevant-authors)
	- [Predicting relevance of articles by relevant authors](#predicting-relevance-of-articles-by-relevant-authors)
- [Submitting Jobs to the Cluster](#submitting-jobs-to-the-cluster)
	- [Submit individual jobs to the cluster](#submit-individual-jobs-to-the-cluster)
	- [Submit a bunch of jobs at once](#submit-a-bunch-of-jobs-at-once)
	- [Deleting multiple jobs](#deleting-multiple-jobs)
	- [Checking successful jobs](#checking-successful-jobs)


## Setup


First, we want to clone this repository on the cluster. Run the following commands on Terminal:

    ssh -p 22022 BC_USERNAME@sirius.bc.edu
	git clone https://github.com/yingyangle/pubmed.git
	cd pubmed

To set up the **conda environment** for our scripts, run the following commands on the cluster from the main `animals` folder after cloning it. I named the environment `tf` but you can name it something else if you want.

    module load anaconda/3-2018.12-P3.7
    conda create -n tf python=3.9.6
    conda activate tf
> **Runtime**: 2 min.

To install the necessary packages, make sure your conda environment is activated and run the following script.

    pip install -r tools/requirements.txt
> **Runtime**: 5 min.
 
You can use the following script to double check that all the package downloads went smoothly.

    python tools/import.py
> **Runtime**: 5 min.

Finally, we want to copy over the files that were too large to be uploaded to GitHub. I changed my folder permissions so you should be able to access my files as long as you're in the `prudhome` user group, but please let me know if you have trouble accessing anything! Make sure to run the following commands from your cloned `pubmed` folder.

    cp /data/yangael/pubmed/data/* data/
    cp /data/yangael/pubmed/bertdata/* bertdata/
    cp /data/yangael/pubmed/saved_models/* saved_models/
    cp /data/yangael/pubmed/results/predictions* results/
    cp /data/yangael/pubmed/PubMed-and-PMC-w2v.bin .
> **Runtime**: a long time

 If you've already set up the environment and files, you can skip most of the previous steps and just make sure to activate the environment before running anything. Also make sure to include this line in your `.pbs` files.
 
 `conda activate tf`
 
 
## Datasets

First, here's an overview of our annotated datasets. All of our annotated data is located in the **`/data`** folder. You'll find our annotated data saved as two types of files:

**`annotations*.csv`** - These files contain only the relevance annotations for each article as well as its PubMed ID and article title. 

**`article_info*.csv`** - These files contain all the metadata for each article, including PubMed ID, article title, abstract, authors, and publication date.

You'll also see numbers such as `1`, `2`, and `1+2` in the filenames for the files mentioned above. These indicate our different batches of annotations.

**`Dataset 1`** was annotated by Prof. Prud'hommeaux's colleagues.

**`Dataset 2`** was annotated by me (Christine).

**`Dataset 1+2`** just combines all the annotations from `Dataset 1` and `Dataset 2`.

Now let's look at the process for annotating articles.

### Article retrieval

First, we need to retrieve a list of articles to annotate. To do this, we query PubMed using the following search terms: `disparity`, `inequity`, and `inequality`. Since our goal is just to annotate 2,000 examples of relevant articles and 2,000 examples of irrelevant articles, we can limit our search to the top 10,000 results for each search term for now. To do this, we can set the `MAX_RESULTS` variable to be `10000`.

    python get_articles.py 'disparity'
    python get_articles.py 'inequity'
    python get_articles.py 'inequality'

The data for the retrieved articles will be saved as `pubmed_articles_*.csv` and `authors_*.json`. 

Then, we can combine all the results from our search terms into one `.csv` file for easy annotating.

    python combine_articles.py

This script combines all the `pubmed_articles_*.csv` files into a single file `to_annotate.csv`, and combined all the `authors_*.json` into `authors.json`. It also excludes any articles that have already been annotated in a given existing annotations file (e.g. `annotations1.csv`). This existing annotations file can be set with the `EXISTING_ANNOTATIONS_FILE` variable.

### Annotation

Here are the labels we use in our annotation:

`0` = irrelevant
`1` = relevant
`3` = unsure

To annotate the data, go through the articles in `to_annotate.csv` and mark 

### Formatting data

After annotating the data, we want to format our data into `annotations*.csv` and `article_info*.csv` files. 

    python split_annotations_info.py

Then we can also combine the annotations from all annotation batches.

    python combine_annotations.py

This will combine `annotations1.csv` and `annotations2.csv` into `annotations1+2.csv`, and does the same thing for the `article_info*.csv` files.


## ML Classification Models

After annotating our data, we can try training a model on our annotations to predict the relevance of a given article based on its title or abstract, or both concatenated together. We can first start with trying some classical machine learning algorithms.

The classification models we test include:
- Logistic Regression
- K Neighbors (k=3)
- Linear SVM
- RBF SVM
- Gaussian Naive Bayes
- Gaussian Process
- Decision Tree
- Random Forest
- Ada Boost
- MLP Neural Net
- Quadratic Discriminant Analysis (QDA)

We can use existing word2vec embeddings trained specifically on PubMed and PMC to convert our text input to vectors. These embeddings are saved in the root folder as `PubMed-and-PMC-w2v.bin`. Our input and output for the models will look something like this:

**Input**: w2v embedding of the article's title, abstract, or both
**Output**: whether or not the article is relevant to health disparities

To train and evaluate our models, we can run the following script:

    python ml.py INPUT_TYPE DATASET

> **Runtime**: 1-7 min.

The evaluation results will be saved in `PubMed_ML_Models.csv` and graphed as `results/w2v_classification*.png`.

## BERT Classification Models

The files for training and evaluating BERT classification models trained using `keras` are located in the root **`pubmed`** folder. 

### Prepare data for training a model

In order to run the scripts to train and evaluate BERT models, we first need to correctly format the data directory for the train and test data by running **`format_bert_data.py`**. This will take each of the `data/annotations*.csv` and `data/article_info*.csv` files and format them as sublists as described in the sample table above. It will automatically create formatted datasets for all the datasets.
				
    python format_bert_data.py DATASET_TYPE
> **Runtime**: 2-15 min. (depending on the script arguments)

Available **`DATASET_TYPE`** options include (you can also run the script with no arguments to see a list of options): 
- **`split`** - prepares data for an **80/20 train test split**
- **`cv`** - prepares data for **5-fold cross validation**
- **`full`** - prepares data for using **all data as training data**
- **`unannotated`** - prepares **unannotated data** for being predicted

If you run the script with  `DATASET_TYPE='unannotated'`, you'll need to add the following two arguments, or run it on the command line rather than submitting a job so that you can be prompted to fill in these variables.

    python format_bert_data.py 'unannotated' ARTICLE_INFO NICKNAME

where `ARTICLE_INFO` is the path for a file containing the article info for the articles you want to include in the data set. The file must contain the PubMed ID, title, and abstract for each article. The `NICKNAME` variable is what you want to name this unannotated dataset (the result will look like `bertdata/bertdata_NICKNAME`).  

The resulting data directory will look something like this:
   

     /bertdata/bertdata_*
        	/train
    	    	/relevant
	    			291.txt
	    			...
	    		/irrelevant
	    			72.txt
	    			...
	    	/test
	    		/relevant
	    			6.txt
	    			...
	    		/irrelevant
	    			103.txt
	    			...

### Train and evaluate a model

Before running the following scripts, make sure you've created the correctly formatted dataset directories as described in [this step](#prepare-data-for-training-a-model).

In the example commands below, I use a number of placeholder variables which you can adjust to be what you want. Here's a summary of what most of the placeholder variables can be set to:

**`DATASET`** = [`1`, `2`, `1+2`]
**`INPUT_TYPE`** = [`title`, `abstract`, `title+abstract`]
**`BERT_MODEL`** = [`bert`, `smallbert`, `albert`, `electra`, `talkingheads`, `experts_pubmed`]

To train a model using an **80/20 train test split**:

    python bert.py BERT_MODEL INPUT_TYPE DATASET

To train a model using **5-fold cross validation**:

    python bert_CV.py BERT_MODEL INPUT_TYPE DATASET NUM_FOLDS

### Evaluate a model on a different dataset

Another way to evaluate how well our model might perform on a different set of data is to train it on one dataset and evaluate it on another. This is useful since the datasets we have are annotated by different people, so we can see how our model handles a slightly different set of data.

    python bert_eval.py BERT_MODEL INPUT_TYPE TRAIN_DATASET TEST_DATASET

You can also run this script with no arguments to try all the different `BERT_MODEL` and `INPUT_TYPE` combinations with all the different `TRAIN_DATASET` and `TEST_DATASET` combinations.

    python bert_eval.py

The results for this script will be saved in `PubMed_BERT_Models_Eval.csv`.

### Using model to generate predictions

Once we've trained some models, we can also use one of these fine-tuned models saved in the `/saved_models`  folder to predict the relevance of some more articles. Before running this, make sure to create the unannotated `bertdata/bertdata*` folder as described [here](#prepare-data-for-training-a-model).

    python bert_predict.py UNANNOTATED_DATASET_NICKNAME


## Searching for Relevant Articles

After training and evaluating to find our best predictive model, we want to move on to steps (4) through (8) of our procedure and use our model to find relevant authors and articles.

### Getting relevant articles

To get a list of all relevant articles, first we want to retrieve all the articles on PubMed matching our search terms (`disparity`, `inequity`, `inequality`), setting the `MAX_RESULTS` limit to be very high (e.g. 1,000,000) so we can get as many articles as PubMed allows.

    python get_articles.py "disparity"
    python get_articles.py "inequity"
    python get_articles.py "inequality"

This script will save the retrieved articles as `data/articles_*.csv`.

After retrieving these articles, we want to predict the relevance of each of these articles using the best model we trained.

    python bert_predict.py BERT_MODEL INPUT_TYPE TRAIN_DATASET UNANNOTATED_DATASET_NICKNAME 

 The results will be saved as `results/predictions_unannotated_*.csv`.


### Getting authors of relevant articles

After getting a list of relevant articles, we want to get a list of the authors of these relevant articles so that we can analyze the trajectory of their research. 

    python get_relevant_authors.py PREDICTIONS_FILE


where `PREDICTIONS_FILE` is the file containing the prediction results (e.g. `'results/predictions_unannotated_*.csv'`).

The results will be saved as  `data/authors_relevant.json`.

### Getting articles by relevant authors

Once we've gotten our list of relevant authors, we want to search for all other articles written by these authors.

    python search_authors.py

This script will take the list of relevant authors in `data/authors_relevant.json` and save the article information for all articles written by each author. The article data will be saved in `data/articles_by_relevant_authors.csv` and `data/articles_by_relevant_authors.json`. 

**Note:** This script might take a long time to run, and you might need to submit multiple jobs to continue the script if you don't set the walltime high enough. If the script does time out, sometimes the `articles_by_relevant_authors.json` file will be corrupted since it wasn't able to finish fully writing the output file before the job timed out. In these cases, you might have to manually fix the file via python on the command line. To make sure it's working, you should be able to load the `.json` file using `json.load()`. After that, you can resubmit the job to continue searching the remaining authors. The script will make sure to skip any authors it has already searched by checking the authors in `articles_by_relevant_authors.json`.

**Another note:** The `articles_by_relevant_authors.json` file might get too big at some point to be loaded in python. You'll get an error that says something like `OSError: [Errno 28] No space left on device`. In this case, rename the `articles_by_relevant_authors.json` file to be something else (e.g. `articles_by_relevant_authors1.json` and create a new blank `articles_by_relevant_authors.json` file just containing `[]`. Do the same for the `articles_by_relevant_authors.csv` file, and create a new blank one with just the header columns as a row of text.

### Predicting relevance of articles by relevant authors

Finally, once we have a full list of all the articles written by our relevant authors, we can run our best predictive model again to predict the relevance of all these articles we found. Before running this, make sure to create the unannotated `bertdata/bertdata*` folder as described [here](#prepare-data-for-training-a-model).

    python bert_predict.py BERT_MODEL INPUT_TYPE TRAIN_DATASET UNANNOTATED_DATASET 

 The results will be saved as `results/predictions_unannotated_*.csv`.


## Submitting Jobs to the Cluster

### Submit individual jobs to the cluster

To submit a job to the cluster, you can edit the `.pbs` files in the `pubmed` folder for convenience (so you don't have to make a bunch of new ones). It's fine to submit multiple jobs to the queue using the same `.pbs` filename, even if the contents of the files are different. 

The **`go.pbs`** and **`misc.pbs`** files contain example commands for running the different scripts, so you can uncomment whichever script you want to run and change the arguments to what you want. Just make sure to update the `walltime` and `mem` settings to be appropriate for the script you're running. 

****Also make sure to change the email in the second line of the `.pbs` file to be your email,** so that you'll receive email notifications instead of me when the job starts and finishes. Or, you can delete the line entirely if you don't want any notifications.

### Submit a bunch of jobs at once 

To make it easier to submit a bunch of jobs to the cluster, I've included a script `tools/write_pbs.py` so that you can mass-submit jobs for a certain script, running through all the dataset and model combinations you want to try. **Make sure to run this script in the main `pubmed` folder like the previous scripts.**

    python tools/write_pbs.py ACTION_TO_RUN

Available options for `ACTION_TO_RUN` include (you can also run the script with no arguments to see a list of options): 
- **`bert_split`** - runs `bert.py` for all models for all datasets and input types
- **`bert_cv`** - runs `bert_CV.py` for all models for all datasets and input types
- **`ml_split`** - runs `ml_models.py` for all datasets and input types

Before running this script, you'll also want to make sure to update the `walltime` and `mem` settings to be what you want. You can do this by editing the `tools/template.pbs` file (e.g. **`nano tools/template.pbs`**). It's usually better to be safe than sorry, so try to set a walltime that you're pretty sure won't time out (although higher walltimes usually also take longer to get to the front of the queue).

### Deleting multiple jobs

Okay sometimes you might realize you accidentally submitted a bunch of jobs using the previous script and there was a typo somewhere in your script. Instead of deleting all these jobs one by one you can use the following script to delete a bunch of consecutive jobs at once:

    python tools/delete_jobs.py FIRST_JOB_ID LAST_JOB_ID

Just put the job ID of the first and last job you want to delete, and the script will delete those and all the jobs with ID number in between them.

### Checking successful jobs

After the jobs complete, you'll get a bunch of output logs like `*.pbs.e*` and `*.pbs.o*`. To make it easier to check the success of these jobs, you can run the following script which will print out the names of jobs that were unsuccessful:

    python tools/check_jobs.py

Most of our scripts print out `RUNTIME: ####` at the very end, so this script just checks the `*.pbs.o*` file to see if it contains this line. Not all scripts have this line though so just double check to see if the script you're checking has this.

