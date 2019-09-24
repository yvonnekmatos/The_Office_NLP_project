from pymongo import MongoClient
from wd import *
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import explained_variance_score
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import cycle, combinations

import warnings
warnings.simplefilter(action='ignore', category=Warning)

# Get all episodes from MongoDB database
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
episode_df.head()
remove_line_breaks = lambda x: re.sub('\r', '', x)
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x)
remove_nums = lambda x: re.sub('\w*\d\w*', '', x)

episode_df['transcript_clean'] = episode_df.transcript.map(remove_line_breaks).map(punc_lower).map(remove_nums)
episode_df.head()

# Get lemmatized version of nouns and proper nouns labeled in Spacy
master_df = pd.read_csv(data_directory+'Spacey_tokenized_data/master_spacy_tokenized.csv')
remaining = pd.read_csv(data_directory+'Spacey_tokenized_data/us_s02_e09_onwards_master_spacy_tokenized.csv')
master_df = master_df.append(remaining)
master_df.shape
master_df.head()

def list_to_string(listed):
    string = ''
    for x in listed:
        string += ' {}'.format(x)
    return string

mask = ((master_df.is_stop==False) & (master_df.pos.isin(['NOUN', 'PROPN'])))
master_filtered = master_df[mask]
master_filtered.head()

one_episode_gr = master_filtered.groupby(['country', 'season', 'episode'])
lemma_dict = {}
for name, group in one_episode_gr:
    lemma_dict[name] = list_to_string(list(group.lemma))


episode_df['for_topic_analysis'] = None
for idx,row in episode_df.iterrows():
    row['for_topic_analysis'] = lemma_dict[(row.country, row.season, row.episode)]

episode_df.shape
episode_df.head()
#==============================================================================================
# FUNCTIONS FOR TOPIC MODLEING

def get_score(model, data, scorer=explained_variance_score):
    """ Estimate performance of the model on the data """
    prediction = model.inverse_transform(model.transform(data))
    return scorer(data, prediction)

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
    print('total: ', sum(lsa.explained_variance_ratio_))
    return doc_topic, display_topics(lsa, vectorizer.get_feature_names(), num_top_words)

def nmf_on_data(vectorizer, uk_transcipts, topics, num_top_words):
    trans_vectorized = vectorizer.fit_transform(uk_transcipts)
    nmf_model = NMF(topics)
    doc_topic = nmf_model.fit_transform(trans_vectorized)
    print('explained variance: ', get_score(nmf_model, trans_vectorized.toarray()))
    return doc_topic, display_topics(nmf_model, vectorizer.get_feature_names(), num_top_words)

def lda_on_data(vectorizer, uk_transcipts, topics, passes):
    trans_vectorized = vectorizer.fit_transform(uk_transcipts).transpose()
    corpus = matutils.Sparse2Corpus(trans_vectorized)
    id2word = dict((v, k) for k, v in vectorizer.vocabulary_.items())
    lda = models.LdaModel(corpus=corpus, num_topics=topics, id2word=id2word, passes=passes)
    return print(lda.print_topics())

#===================================================================================================================
# TOPIC MODELNG, US and UK separate, leave in characters
# This analysis treating US and UK separately is a proof of concept - will the topics make sense?

episode_df['for_topic_analysis'] = episode_df.for_topic_analysis.str.lower()
episode_df.head()

# Take all 14 UK episodes
uk = episode_df[episode_df.country=='UK']
uk_transcipts = uk.for_topic_analysis
len(uk_transcipts)

# Take first 14 US episodes
us = episode_df[episode_df.country=='US'].iloc[:14, :]
us_transcipts = us.for_topic_analysis
len(us_transcipts)

exclude = ['okay', 'ok', 'um', 'uh', 'time', 'yeah', 'yep', 'lets', 'hey', 'right', 'gonna', \
           'mr', 'guy', 'good', 'office', 'job', 'sort', 'god', 'work', 'job', 'girl', 'year', 'minute', \
           'hour', 'everybody', 'woman', 'life', 'company', 'line', 'idea', 'meeting', 'word',\
           'ill', 'bit', 'day', 'thay', 'stuff', 'fine', 'mans', 'way', 'people', \
           'thanks', 'questions', 'dundie', 'googi', 'man', 'thing', \
            'night', 'tonight', 'today', 'question', 'answer', 'head', 'laugh', \
            'home', 'cool', 'mile', 'feelin', 'team', 'money', 'paper', 'friend', 'point', 'leg', \
            'oink', 'thank', 'lot', 'face', 'problem', \
            'stuart', 'rogers', 'boy', 'warmer', 'yup']

stop_words = text.ENGLISH_STOP_WORDS.union(exclude)
vectorizer = CountVectorizer(stop_words=stop_words)
tfidf = TfidfVectorizer(stop_words=stop_words)


corpuses = {'US': us_transcipts, 'UK': uk_transcipts}
vectorizer_types = {'CountVectorizer': vectorizer, 'TF-IDF': tfidf}

# CountVectorizer with LSA or NMF both perform well
for c_k, c_v in corpuses.items():
    for v_k, v_v in vectorizer_types.items():
        print('{}, {}, LSA'.format(c_k, v_k))
        lsa_on_data(v_v, c_v, topics=8, num_top_words=5)
        print('')
        print('{}, {}, NMF'.format(c_k, v_k))
        nmf_on_data(v_v, c_v, topics=8, num_top_words=5)
        print('')
        print('{}, {}, LDA'.format(c_k, v_k))
        lda_on_data(v_v, c_v, topics=8, passes=5)
        print('')

# CountVectorizer seems to work best with either LSA or NMF
# Topics are making sense so far - will only try using bi-grams or tri-grams if necessary

#======================================================================================================
#======================================================================================================
#======================================================================================================
# TOPIC MODELNG, single US and UK corpus, remove character names

uk_us = episode_df.iloc[:28, :]
uk_us['country_digit'] = uk_us.apply(lambda row: 0 if row.country=='UK' else 1, axis=1)
uk_us.head()

exclude_char = ['okay', 'ok', 'um', 'uh', 'time', 'yeah', 'yep', 'lets', 'hey', 'right', 'gonna', \
           'mr', 'guy', 'good', 'office', 'job', 'sort', 'god', 'work', 'job', 'girl', 'year', 'minute', \
           'hour', 'everybody', 'woman', 'life', 'company', 'line', 'idea', 'meeting', 'word',\
           'ill', 'bit', 'day', 'thay', 'stuff', 'fine', 'mans', 'way', 'people', \
           'thanks', 'questions', 'dundie', 'googi', 'man', 'thing', \
            'night', 'tonight', 'today', 'question', 'answer', 'head', 'laugh', \
            'home', 'cool', 'mile', 'feelin', 'team', 'money', 'paper', 'friend', 'point', 'leg', \
            'oink', 'thank', 'lot', 'face', 'problem', \
            'stuart', 'rogers', 'boy', 'warmer', 'yup', \
            'michael', 'scott', 'dwight', 'schrute', 'jim', 'halpert', \
            'pam', 'beesly', 'ryan', 'howard', 'jan', 'levinson', 'gould', \
            'roy', 'anderson', 'stanley', 'hudson', 'kevin', 'malone', \
            'meredith', 'palmer', 'angela', 'martin', 'oscar', 'martinez', \
            'phyllis', 'lapin', 'kelly', 'kapoor', 'toby', 'flenderson', \
            'creed', 'bratton', 'darryl', 'philbin', 'todd', 'packer', \
            'david', 'wallace', 'katy',\
            'brent', 'tim', 'canterbury', 'gareth', 'keenan', 'dawn', 'tinsley', \
            'jennifer', 'taylor', 'clarke', 'ricky', 'howard', 'chris', 'finch', \
            'neil', 'godwin', 'rachel', 'anne', 'keith', 'bishop', 'lee', 'glynn', \
            'malcolm', 'donna', 'karen', 'roper', 'trudy', 'oliver', 'brenda', \
            'rowan', 'simon', 'ray', 'helena', 'oggy', 'nathan', 'jude', 'simon', \
            'jimmy', 'tony',
            ]
stop_words_no_char = text.ENGLISH_STOP_WORDS.union(exclude_char)
vectorizer_nc = CountVectorizer(stop_words=stop_words_no_char)
tfidf_nc = TfidfVectorizer(stop_words=stop_words_no_char)

vectorizer_types_nc = {'CountVectorizer': vectorizer_nc, 'TF-IDF': tfidf_nc}

# Try different vectorizers and topic modeling with LSA, NMF, and LDA
for v_k, v_v in vectorizer_types_nc.items():
    print('{}, LSA'.format(v_k))
    lsa_on_data(v_v, uk_us.for_topic_analysis, topics=16, num_top_words=5)
    print('')
    print('{}, NMF'.format(v_k))
    nmf_on_data(v_v, uk_us.for_topic_analysis, topics=16, num_top_words=5)
    print('')
    print('{}, LDA'.format(v_k))
    lda_on_data(v_v, uk_us.for_topic_analysis, topics=16, passes=5)
    print('')

# NMF with CountVectorizer performs the best, next try to increase number of topics and get variance up
topics = range(16, 21)

for topic in topics:
    print('num topics: ', topic)
    nmf_matrix, nmf_topics = nmf_on_data(vectorizer_nc, uk_us.for_topic_analysis, topics=topic, num_top_words=5)
    print('')

# Best topic model - 20 topics with CountVectorizer and NMF, gives high explained variance and topics make sense
# topics are very similar to when you look at us and uk separately

nmf_matrix_nc, nmf_topics_nc = nmf_on_data(vectorizer_nc, uk_us.for_topic_analysis, topics=20, num_top_words=5)
nmf_matrix_nc

#======================================================================================================
# ANALYZE TOPICS - NORMAL VS CRAZY
# Overall Question: Is the US version wackier/crazier?
# Data Question: Which country has a higher proportion of crazy topics?

# Label topics from NMF matrix as normal (N) or crazy (C)
# A normal topic is something that would come up multiple times throughout the show
    # Example normal topic (picture, room, redundancy, warehouse, boss)
# A crazy topic is somthing out of the ordinary that happened once or hardly at all
    # Example crazy topic (sensei, contact, fight, belt, emergency)

topic_dict = {0: ['N', 'UK']
              , 1: ['N', 'US']
              , 2: ['C', 'US']
              , 3: ['N', 'US']
              , 4: ['C', 'both']
              , 5: ['C', 'UK']
              , 6: ['N', 'US']
              , 7: ['C', 'UK']
              , 8: ['C', 'US']
              , 9: ['C', 'US']
              , 10: ['N', 'UK']
              , 11: ['C', 'US']
              , 12: ['N', 'US']
              , 13: ['N', 'UK']
              , 14: ['C', 'US']
              , 15: ['N', 'UK']
              , 16: ['C', 'US']
              , 17: ['N', 'both']
              , 18: ['C', 'US']
              , 19: ['N', 'UK']
              }

topic_df = pd.DataFrame.from_dict(topic_dict).T.reset_index()
topic_df.columns = ['topic', 'normal_crazy', 'topic_country']


nmf_df = pd.DataFrame(nmf_matrix_nc)
uk_nmf = nmf_df.iloc[:14,:]
us_nmf = nmf_df.iloc[14:,:]

topic_df_uk = topic_df.copy()
topic_df_uk['topic_mean'] = uk_nmf.mean()
topic_df_uk['episode_country'] = 'UK'

topic_df_us = topic_df.copy()
topic_df_us['topic_mean'] = us_nmf.mean()
topic_df_us['episode_country'] = 'US'

topic_df_all = topic_df_uk.append(topic_df_us)

colors = {"UK": "#6a79f7", "US": "#1fa774"}
sns.catplot(x="normal_crazy", y="topic_mean", hue="episode_country", data=topic_df_all,
            kind="bar", palette=colors, height=6, ci=None).savefig("data_viz/topic_bars3.png", bbox_inches='tight')

            .savefig("data_viz/topic_bars3.png", bbox_inches='tight')

#======================================================================================================
# TOPIC ANALYSIS
# Overall Question: Is the UK version more consistent? Does office life generally remain the same throughout?
# Data Question: Which episode pairs are most similar to each other?

Vt = pd.DataFrame(nmf_matrix_nc.round(5))
Vt.shape

# Find cosine similarity of all unique pairs of episodes
mt = np.array([[1,1], [1,1]])
id = np.array([[1,0], [0,1]])

similar_epiodes = []
dissimilar_episodes = []

combi = list(combinations(np.arange(0, 28), 2))
print(len(combi))
for comb in combi:
    cos_sim = cosine_similarity((Vt.iloc[comb[0]], Vt.iloc[comb[1]])).round()
    print(cos_sim)
    # episodes that have a cosine_similarity of >= 0.5
    if np.array_equal(cos_sim, mt):
        similar_epiodes.append(comb)
    # episodes that have a cosine_similarity of < 0.5
    elif np.array_equal(cos_sim, id):
        dissimilar_episodes.append(comb)


sim_ranking = []
for episode in similar_epiodes:
    cos_sim = cosine_similarity((Vt.iloc[episode[0]], Vt.iloc[episode[1]])).round(2)
    sim_ranking.append((episode, cos_sim[0][1]))

sim_ranking.sort(key=lambda x: x[1], reverse=True)

ranking_df = pd.DataFrame()
ranking_df['rank_order'] = list(np.arange(0,17))
ranking_df['episode_pair'] = [x[0] for x in sim_ranking]
ranking_df['cosine_similarity'] = [x[1] for x in sim_ranking]
ranking_df

# Plot cosine similarities >= 0.5
x_vals = ranking_df.rank_order.values
y_vals = ranking_df.cosine_similarity.values

uk_us_pairs = [0,6,12]
uk_only_pairs = [1,2,3,4,5,7,8,9,10,11,13,16]
us_only_pairs = [14,15]

barplot = plt.bar(x_vals, y_vals, align='center', alpha=1)
plt.xticks(y_pos, objects)
plt.ylim([0.5,1])
plt.grid(False)
for uk_us_pair in uk_us_pairs:
    barplot[uk_us_pair].set_color('#ab1239')
for uk_only_pair in uk_only_pairs:
    barplot[uk_only_pair].set_color('#6a79f7')
for us_only_pair in us_only_pairs:
    barplot[us_only_pair].set_color('#1fa774')
plt.savefig("data_viz/cos_sim_bars2.png", bbox_inches='tight', transparent=True)
plt.show()
