from pymongo import MongoClient
from pprint import pprint
from os import listdir
from wd import *
import requests
from copy import deepcopy
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from nlp_pipeline import *
import spacy

import warnings
warnings.simplefilter(action='ignore', category=Warning)

# notes: do topic modeling first, then add in ngrams into the CountVectorizer if the simple topic modeling isnt working

client = MongoClient()

client.list_database_names()

office_db = client['office']

all_episodes = office_db.get_collection('all_episodes')

all_episodes.count()

cursor = all_episodes.find({}, {'_id': 0}).limit(28)
episode_df = pd.DataFrame()
for elem in cursor:
    df = pd.DataFrame.from_dict(elem, orient='index').T # columns=['country', 'season', 'episode', 'transcript']
    episode_df = episode_df.append(df)

episode_df = episode_df.reset_index(drop=True)
episode_df


remove_line_breaks = lambda x: re.sub('\r', '', x)
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x) #, x.lower()
remove_nums = lambda x: re.sub('\w*\d\w*', '', x)

episode_df['transcript_clean'] = episode_df.transcript.map(remove_line_breaks).map(punc_lower).map(remove_nums)

#========================================================================================
# TRY SPACY
episode_df.head()
episode_df.transcript_clean.values[:4]

nlp = spacy.load('en_core_web_sm')
doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')
episode_df.iloc[0:4,:]

master_df = pd.DataFrame()
for idx,row in episode_df.iterrows(): # enumerate(episode_df)
    df = pd.DataFrame()
    token_dict = {}
    doc = nlp(row.transcript_clean)
    for token in doc:
        token_list = []
        attributes = [token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                token.shape_, token.is_alpha, token.is_stop]
        for att in attributes:
            token_list.append(att)
            token_dict[(row.country, row.season, row.episode)] = token_list
            df2 = pd.DataFrame.from_dict(token_dict).T.reset_index()
        df = df.append(df2)
    df.to_csv('spacy_tokenized_{}_{}_{}'.format(row.country, row.season, row.episode), index=False)
    master_df = master_df.append(df)

master_df.columns = ['country', 'season', 'episode', 'text', 'lemma', 'pos', 'tag', 'dep',
               'shape', 'is_apha', 'is_stop']
master_df = master_df.reset_index(drop=True)
# master_df
master_df.to_csv('master_spacy_tokenized.csv', index=False)



#========================================================================================
# TEST
def simple_cleaning_function_i_made(text, tokenizer, stemmer):
    cleaned_text = []
    for post in text:
        cleaned_words = []
        for word in tokenizer(post):
            low_word = word.lower()
            if stemmer:
                low_word = stemmer.stem(low_word)
            cleaned_words.append(low_word)
        cleaned_text.append(' '.join(cleaned_words))
    return cleaned_text


def lsa_on_data_transformed(trans_vectorized, topics, num_top_words):
    lsa = TruncatedSVD(topics)
    doc_topic = lsa.fit_transform(trans_vectorized)
    print('explained variance ratio: ', lsa.explained_variance_ratio_)
    return display_topics(lsa, vectorizer.get_feature_names(), num_top_words)


from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer

nlp = nlp_preprocessor(vectorizer=CountVectorizer(), cleaning_function=simple_cleaning_function_i_made,
                       tokenizer=TreebankWordTokenizer().tokenize, stemmer=PorterStemmer())

nlp.fit(us.transcript_clean)
a = nlp.transform(us.transcript_clean).toarray()

lsa_on_data_transformed(a, topics=8, num_top_words=5)
