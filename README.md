# PubMed Health Disparity

** **(This README is a WIP)** **

## Setup

First, we want to clone this repository on the cluster.

    ssh -p 22022 BC_USERNAME@sirius.bc.edu
    git clone https://github.com/yingyangle/pubmed.git
    cd pubmed



To set up the **conda environment** for our scripts, run the following commands on the cluster from the main `pubmed` folder after cloning it. I named the environment `tf` but you can name it something else if you want.

    module load anaconda/3-2018.12-P3.7
    conda create tf
    conda activate tf
    pip install -r tools/requirements.txt


## BERT Classification Models

The files for training and evaluating BERT classification models trained using `keras` are contained in the **`pubmed`** folder. 

### Prepare data for training a model

To prepare data for an **80/20 train test split**:
**`python format_bert_data.py DATASET`**
e.g. `python format_bert_data.py 'mturk'`

To prepare data for **5-fold cross validation**:
**`python format_bert_data_CV.py DATASET NUM_FOLDS`**
e.g. `python format_bert_data.py 'mturk' '5'`


### Train and evaluate a model

Before running the following scripts, make sure you've created the correctly formatted dataset directories as described in [this step](#prepare-data-for-training-a-model).

To train a model using an **80/20 train test split**:
**`python bert.py DATASET MODEL_NAME`**
e.g. `python bert.py 'mturk' 'smallbert'`

To train a model using **5-fold cross validation**:
**`python bert.py DATASET MODEL_NAME NUM_FOLDS`**
e.g. `python bert_CV.py 'mturk' 'electra' '5'`

### Evaluate a model on a different dataset

**`python bert_eval.py BERT_MODEL TRAIN_DATASET`**
e.g. `python bert_eval.py 'talkingheads' 'mturk'`

### Submit individual jobs to the cluster

To submit a job to the cluster, you can edit the `.pbs` files in the `pubmed` folder for convenience (so you don't have to make a bunch of new ones). It's fine to submit multiple jobs to the queue using the same `.pbs` filename, even if the contents of the files are different. 

The **`tier.pbs`** file contains example commands for running the different scripts, so you can uncomment whichever script you want to run and change the arguments to what you want. Just make sure to update the `walltime` and `mem` settings to be appropriate for the script you're running. 

****Also make sure to change the email in the second line of the `.pbs` file to be your email,** so that you'll receive email notifications instead of me when the job starts and finishes. Or, you can delete the line entirely if you don't want any notifications.

### Submit a bunch of jobs at once 

To make it easier to submit a bunch of jobs to the cluster, I've included a script `tools/write_pbs.py` so that you can mass-submit jobs for a certain script, running through all the dataset and model combinations you want to try. **Make sure to run this script in the main `pubmed` folder like the previous scripts.**

**`python tools/write_pbs.py ACTION_TO_RUN`**
e.g. `python tools/write_pbs.py 'run_bert'`

Available actions include (you can also run the script with no arguments to see a list of options): 
- **`run_bert`** - runs `bert.py` for all models for all datasets
- **`run_bert_CV`** - runs `bert_CV.py` for all models for all datasets

Before running this script, you'll also want to make sure to update the `walltime` and `mem` settings to be what you want. You can do this by editing the `tools/template.pbs` file (e.g. **`nano tools/template.pbs`**). It's usually better to be safe than sorry, so try to set a walltime that you're pretty sure won't time out (although higher walltimes usually also take longer to get to the front of the queue).
