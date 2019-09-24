from pymongo import MongoClient
from wd import *
import pandas as pd
import numpy as np
import re
import string
import spacy

import warnings
warnings.simplefilter(action='ignore', category=Warning)


client = MongoClient()
client.list_database_names()
office_db = client['office']
all_episodes = office_db.get_collection('all_episodes')
all_episodes.count()

cursor = all_episodes.find({}, {'_id': 0})
episode_df = pd.DataFrame()
for elem in cursor:
    df = pd.DataFrame.from_dict(elem, orient='index').T # columns=['country', 'season', 'episode', 'transcript']
    episode_df = episode_df.append(df)

episode_df = episode_df.reset_index(drop=True)
episode_df


remove_line_breaks = lambda x: re.sub('\r', '', x)
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x)
remove_nums = lambda x: re.sub('\w*\d\w*', '', x)

episode_df['transcript_clean'] = episode_df.transcript.map(remove_line_breaks).map(punc_lower).map(remove_nums)

#========================================================================================
# GET LEAMMATIZED WORDS AND PARTS OF SPEECH WITH SPACY

episode_df.tail()
episode_df.transcript_clean.values[:4]

nlp = spacy.load('en_core_web_sm')

episode_df.iloc[28:,:]


master_df = pd.DataFrame()
for idx,row in episode_df.iloc[28:,:].iterrows():
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

master_df.to_csv('us_s02_e09_onwards_master_spacy_tokenized.csv', index=False)
