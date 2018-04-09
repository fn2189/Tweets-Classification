# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 11:58:26 2018

@author: franck
"""
import pandas as pd
import numpy as np
import pickle
import argparse
from sklearn.externals import joblib

import re
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk import pos_tag, ne_chunk
import pickle
from sklearn.externals import joblib
import os
os.environ['JAVA_HOME'] = 'C:/Program Files (x86)/Java/jdk1.8.0_151'

import string

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb


class NLTKPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, stopwords=None, punct=None,
                 lower=True, strip=True, remove_stopwords=True):
        self.lower      = lower
        self.strip      = strip
        self.remove_stopwords = remove_stopwords
        self.stopwords  = stopwords or set(sw.words('english'))
        self.punct      = punct or set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()
        self.dates = [
        'mon(?![a-z])', 'monday',
        'tues(?![a-z])', 'tuesday',
        'wednesday',
        'thur(?![a-z])', 'thurs', 'thursday',
        'fri(?![a-z])', 'friday',
        'saturday',
        'sunday'
    ]

    def fit(self, X, y=None):
        return self

    def join(self, X):
        return [" ".join(doc) for doc in X]

    def transform(self, X):
        l =  [
            list(self.tokenize(doc)) for doc in X
        ]
        return self.join(l)

    def tokenize(self, document):
        # Break the document into sentences
        for sent in sent_tokenize(document):
            # Break the sentence into part of speech tagged tokens
            sent_acc = self.replace_accounts(sent)
            sent_links = self.replace_links(sent_acc)
            sent_cities = self.replace_cities(sent_links)
            sent_times = self.replace_times(sent_cities)
            sent_delays = self.replace_delays(sent_times)
            sent_dates = self.replace_dates(sent_delays)
            sent_hashtags = self.replace_hashtags(sent_dates)
            for token, tag in pos_tag(wordpunct_tokenize(sent_hashtags)):
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token
    
                # If stopword, ignore token and continue
                if self.remove_stopwords:
                    if token in self.stopwords:
                        continue

                # If punctuation, ignore token and continue
                if all(char in self.punct for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                yield lemma
                
    def replace_accounts(self, sen):
        return re.sub('@[^ ]+', 'TOPERSON0', sen)
    
    def replace_links(self, sen):
        return re.sub('http[:][/]{2}[^ ]+', '@LINK0', sen)
    
    def replace_cities(self, sen):
        return re.sub('(?<![A-Z0-9])([A-Z]{2,3})(?![A-Z0-9])', '@CITY0', sen)
    
    def replace_times(self, sen):
        return re.sub('[0-9]{1,2}[:][0-9]{1,2}', ' TIME0 ', sen)
    def replace_delays(self, sen):
        return re.sub('[0-9]{1,2}[ ]*(minutes|minute|mins|min|hours|hours|hrs|hr)', 'DELAY0', sen)
    
    
    def replace_dates(self, sen):
        return re.sub('('+'|'.join(self.dates)+')', 'DATE0', sen, flags=re.IGNORECASE)
    
    def replace_hashtags(self, sen):
        return re.sub('#[a-zA-Z0-9]+', 'HASHTAG0', sen)
    

    def lemmatize(self, token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

def predict(tweet, model_files='../saved_models/'):
    """
    Makes a prediction on the user specified tweet using 
    the passed model
    """
    pipe = joblib.load(model_files + 'pipe.pkl')
    final_svm = joblib.load(model_files + 'final_svm.pkl')
    final_logit = joblib.load(model_files + 'final_logit.pkl')
    final_boost = joblib.load(model_files + 'final_boost.pkl')
    final_LGB = joblib.load(model_files + 'final_LGB.pkl')
    final_estimator = joblib.load(model_files + 'final_estimator.pkl')
    to_predict = [tweet]
    X_test = pipe.transform(to_predict).astype(np.float64)
    columns = ['SVM', 'Boost', 'Logit', 'LGB']
    test_meta = pd.DataFrame(np.nan, index=range(X_test.shape[0]), columns=columns)
    test_meta['SVM'] = final_svm.predict_proba(X_test)[:,:1].flatten()
    test_meta['Logit'] = final_logit.predict_proba(X_test)[:,1].flatten()
    test_meta['Boost'] = final_boost.predict_proba(X_test)[:,1].flatten()         
    test_meta['LGB'] = final_LGB.predict_proba(X_test)[:,1].flatten()
    res = final_estimator.predict_proba(test_meta)
    #print(tweet)
    print('Proability of this tweet being related to a late flight :',
        res[0][1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tweet", type=str,
                    help="the tweet for prediction")
    
    args = parser.parse_args()
    predict(args.tweet)
    
if __name__ == '__main__':
    main()