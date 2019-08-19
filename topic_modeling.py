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
episode_df.head()
remove_line_breaks = lambda x: re.sub('\r', '', x)
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x) #, x.lower()
remove_nums = lambda x: re.sub('\w*\d\w*', '', x)

episode_df['transcript_clean'] = episode_df.transcript.map(remove_line_breaks).map(punc_lower).map(remove_nums)
episode_df.head()

master_df = pd.read_csv(data_directory+'Spacey_tokenized_data/master_spacy_tokenized.csv')
master_df.shape
master_df.head(10)


mask = ((master_df.is_stop==False) & (master_df.pos.isin(['NOUN', 'PROPN'])))
master_filtered = master_df[mask]
master_filtered

test_gr = master_filtered.groupby(['country', 'season', 'episode'])
test_dict = {}
for name, group in test_gr:
    test_dict[name] = list(group.text)


def filter_words(text_list, transcript):
    string = ''
    trans_split = transcript.split(' ')
    for word in trans_split:
        if word in text_list:
            string += ' {}'.format(word)
    return string

episode_df['for_topic_analysis'] = None
for idx,row in episode_df.iterrows():
    test_list = test_dict[(row.country, row.season, row.episode)]
    row['for_topic_analysis'] = filter_words(test_list, row.transcript_clean)

# episode_df


episode_df.shape
episode_df.head()
#==============================================================================================
def display_topics(model, feature_names, no_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

def lsa_on_data(vectorizer, uk_transcipts, topics, num_top_words):
    trans_vectorized = vectorizer.fit_transform(uk_transcipts)
    lsa = TruncatedSVD(topics)
    doc_topic = lsa.fit_transform(trans_vectorized)
    print('explained variance ratio: ', lsa.explained_variance_ratio_)
    return display_topics(lsa, vectorizer.get_feature_names(), num_top_words)


episode_df['for_topic_analysis'] = episode_df.for_topic_analysis.str.lower()


uk = episode_df[episode_df.country=='UK']
uk_transcipts = uk.for_topic_analysis

us = episode_df[episode_df.country=='US']
us_transcipts = us.for_topic_analysis

exclude = ['okay', 'ok', 'um', 'uh', 'time', 'yeah', 'yep', 'lets', 'hey', 'right', 'gonna', \
           'mr', 'guy', 'good', \
           'ill', 'bit', 'day', 'thay', 'stuff', 'fine', 'mans', 'way', 'people', \
           'thanks', 'questions', 'dundie']
stop_words = text.ENGLISH_STOP_WORDS.union(exclude)
vectorizer = CountVectorizer(stop_words=stop_words) # , max_df=0.9, min_df=0.1
lsa_on_data(vectorizer, us_transcipts, topics=8, num_top_words=5)

lsa_on_data(vectorizer, uk_transcipts, topics=8, num_top_words=5)







trans_vectorized = vectorizer.fit_transform(uk_transcipts)

pd.DataFrame(trans_vectorized.toarray(), columns=vectorizer.get_feature_names(), index=uk.episode.values).head().T

lsa = TruncatedSVD(8)
doc_topic = lsa.fit_transform(trans_vectorized)
lsa.explained_variance_ratio_

topic_word = pd.DataFrame(lsa.components_.round(3),
             index = ["component_1","component_2", "component_3","component_4"],
             columns = vectorizer.get_feature_names())
topic_word


display_topics(lsa, vectorizer.get_feature_names(), 5)




# transcipts

stop_words = text.ENGLISH_STOP_WORDS.union(['ok', 'okay', 'um', 'uh', 'gonna'])
vectorizer = CountVectorizer(stop_words=stop_words, max_df=0.9, min_df=0.1)

lsa_on_data(vectorizer, us_transcipts, topics=8, num_top_words=5)

# tfidf = TfidfVectorizer(stop_words='english', max_df=0.93, min_df=0.1)
stop_words = text.ENGLISH_STOP_WORDS.union(['ok', 'okay', 'um', 'uh', 'gonna'])
tfidf = TfidfVectorizer(stop_words=stop_words, max_df=0.93, min_df=0.1)
lsa_on_data(tfidf, us_transcipts, topics=8, num_top_words=5)
# CREATE WORD VECTORS
# count vectorizer
# TF-IDF
# word embeddings

# TOPIC MODELING
# LSA
# LDA
# NMF
# word embeddings?
