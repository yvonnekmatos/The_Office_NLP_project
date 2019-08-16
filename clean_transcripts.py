from pymongo import MongoClient
from pprint import pprint
from os import listdir
from wd import *
import requests
from copy import deepcopy
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

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
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x.lower())
remove_nums = lambda x: re.sub('\w*\d\w*', '', x)

episode_df['transcript_clean'] = episode_df.transcript.map(remove_line_breaks).map(punc_lower).map(remove_nums)

episode_df

uk = episode_df[episode_df.country=='UK']

uk_transcipts = uk.transcript_clean
# transcipts

vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=0.1)



trans_vectorized = vectorizer.fit_transform(uk_transcipts)

pd.DataFrame(trans_vectorized.toarray(), columns=vectorizer.get_feature_names(), index=uk.episode.values).head().T

lsa = TruncatedSVD(8)
doc_topic = lsa.fit_transform(trans_vectorized)
lsa.explained_variance_ratio_

topic_word = pd.DataFrame(lsa.components_.round(3),
             index = ["component_1","component_2", "component_3","component_4"],
             columns = vectorizer.get_feature_names())
topic_word

def display_topics(model, feature_names, no_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

display_topics(lsa, vectorizer.get_feature_names(), 5)



#==========================================================================================================
# US

def lsa_on_data(vectorizer, uk_transcipts, topics, num_top_words):
    trans_vectorized = vectorizer.fit_transform(uk_transcipts)
    lsa = TruncatedSVD(topics)
    doc_topic = lsa.fit_transform(trans_vectorized)
    lsa.explained_variance_ratio_
    return display_topics(lsa, vectorizer.get_feature_names(), num_top_words)

us = episode_df[episode_df.country=='US']

us_transcipts = us.transcript_clean
# transcipts

vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=0.1)


lsa_on_data(vectorizer, us_transcipts, topics=8, num_top_words=5)

# CREATE WORD VECTORS
# count vectorizer
# TF-IDF
# word embeddings

# TOPIC MODELING
# LSA
# LDA
# NMF
# word embeddings?
