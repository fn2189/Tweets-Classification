# Tweets-Classification
Tweet classifier for airlines


## Data

The data consisted of pairs of tweet and label. There is also some metadata that was discarded for this specific project.
Initially the data contained multiple categories or the label, but I restrained the proble to a binary classification between tweets that are related to a late flight and tweets that are not.
owever, this general method could be applied more or less as is for a multiclass problem

## Notebook

The notebook contains the walkthrought of the differents steps I went through to tune the model.

## Main/predict

The predict file is the python command-line script that make prediction for a new tweet passed as an argument.
Use python predict.py --help for how to use it.
