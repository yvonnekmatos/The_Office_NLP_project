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
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import explained_variance_score
from nlp_pipeline import *
import spacy
import gensim
from gensim import corpora, models, similarities, matutils
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from itertools import cycle, combinations

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
episode_df.head()
remove_line_breaks = lambda x: re.sub('\r', '', x)
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x) #, x.lower()
remove_nums = lambda x: re.sub('\w*\d\w*', '', x)

episode_df['transcript_clean'] = episode_df.transcript.map(remove_line_breaks).map(punc_lower).map(remove_nums)
episode_df

master_df = pd.read_csv(data_directory+'Spacey_tokenized_data/master_spacy_tokenized.csv')
remaining = pd.read_csv(data_directory+'Spacey_tokenized_data/us_s02_e09_onwards_master_spacy_tokenized.csv')
master_df = master_df.append(remaining)
master_df.shape
master_df

def list_to_string(listed):
    string = ''
    for x in listed:
        string += ' {}'.format(x)
    return string

mask = ((master_df.is_stop==False) & (master_df.pos.isin(['NOUN', 'PROPN'])))
master_filtered = master_df[mask]
master_filtered

test_gr = master_filtered.groupby(['country', 'season', 'episode'])
test_dict = {}
for name, group in test_gr:
    test_dict[name] = list_to_string(list(group.lemma))


episode_df['for_topic_analysis'] = None
for idx,row in episode_df.iterrows():
    row['for_topic_analysis'] = test_dict[(row.country, row.season, row.episode)]


episode_df['new_col'] = episode_df.transcript_clean.apply(list_to_string)
episode_df.shape
episode_df.head()
#==============================================================================================
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

episode_df['for_topic_analysis'] = episode_df.for_topic_analysis.str.lower()
episode_df
#===================================================================================================================
# TOPIC MODELNG, US and UK separate, leave in characters
uk = episode_df[episode_df.country=='UK']
uk_transcipts = uk.for_topic_analysis

us = episode_df[episode_df.country=='US'].iloc[:14, :]
us_transcipts = us.for_topic_analysis


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
# Bigrams did not help
# cv2 = CountVectorizer(ngram_range=(1,2), binary=True, stop_words=stop_words)
# tfidf2 = TfidfVectorizer(ngram_range=(1,2), binary=True, stop_words=stop_words)


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




















#======================================================================================================
#======================================================================================================
#======================================================================================================
# PCA DID NOT WORK AS WELL!

# PCA (topic modeling first, combining us and uk, leave in characters)
# topics are very similar to when you look at us and uk separately

def plot_PCA_2D_full(data, target, target_names, save_name):
    colors = cycle(['r','b','g', 'c','m','y','orange','w','aqua','yellow'])
    target_ids = range(len(target_names))
    plt.figure(figsize=(10,10))
    for i, c, label in zip(target_ids, colors, target_names):
        plt.scatter(data[target == i, 0], data[target == i, 1],
                   c=c, label=label, edgecolors='gray')
    plt.legend()
    plt.savefig('{}.png'.format(save_name), bbox_inches='tight')

def plot_PCA_2D_zoomed(data, target, target_names, xlim, ylim, save_name):
    colors = cycle(['r','b','g', 'c','m','y','orange','w','aqua','yellow'])
    target_ids = range(len(target_names))
    plt.figure(figsize=(10,10))
    plt.xlim(xlim)
    plt.ylim(ylim)
    for i, c, label in zip(target_ids, colors, target_names):
        plt.scatter(data[target == i, 0], data[target == i, 1],
                   c=c, label=label, edgecolors='gray')
    plt.legend()
    plt.savefig('{}.png'.format(save_name), bbox_inches='tight')


uk_us = episode_df.iloc[:28, :]
uk_us['country_digit'] = uk_us.apply(lambda row: 0 if row.country=='UK' else 1, axis=1)

vectorizer = CountVectorizer(stop_words=stop_words)
tfidf = TfidfVectorizer(stop_words=stop_words)

vectorizer_types = {'CountVectorizer': vectorizer, 'TF-IDF': tfidf}

# NMF with CountVectorizer performs the best, next try to increase number of topics and get variance up
for v_k, v_v in vectorizer_types.items():
    print('{}, LSA'.format(v_k))
    lsa_on_data(v_v, uk_us.for_topic_analysis, topics=16, num_top_words=5)
    print('')
    print('{}, NMF'.format(v_k))
    nmf_on_data(v_v, uk_us.for_topic_analysis, topics=16, num_top_words=5)
    print('')
    print('{}, LDA'.format(v_k))
    lda_on_data(v_v, uk_us.for_topic_analysis, topics=16, passes=5)
    print('')

topics = range(16, 21)

for topic in topics:
    print('num topics: ', topic)
    nmf_matrix, nmf_topics = nmf_on_data(vectorizer, uk_us.for_topic_analysis, topics=topic, num_top_words=5)

nmf_matrix, nmf_topics = nmf_on_data(vectorizer, uk_us.for_topic_analysis, topics=19, num_top_words=5)

pca = PCA(n_components=2)
pca.fit(nmf_matrix)
pcafeatures_train = pca.transform(nmf_matrix)
target_names = uk_us.country_digit.values


plot_PCA_2D_full(pcafeatures_train, target=target_names, target_names=['UK', 'US'], save_name='data_viz/char_full')


plot_PCA_2D_zoomed(pcafeatures_train, target=target_names, target_names=['UK', 'US'], xlim=[-1,1.5], ylim=[-1.5,1], save_name='data_viz/char_zoomed')
#======================================================================================================
# PCA (topic modeling first, combining us and uk, REMOVE characters)

uk_us

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

# NMF with CountVectorizer performs the best, next try to increase number of topics and get variance up
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

topics = range(16, 21)

for topic in topics:
    print('num topics: ', topic)
    nmf_matrix, nmf_topics = nmf_on_data(vectorizer_nc, uk_us.for_topic_analysis, topics=topic, num_top_words=5)
    print('')

# Best topic model
nmf_matrix_nc, nmf_topics_nc = nmf_on_data(vectorizer_nc, uk_us.for_topic_analysis, topics=20, num_top_words=5)
nmf_matrix_nc

pca_nc = PCA(n_components=2)
pca_nc.fit(nmf_matrix_nc)
pcafeatures_train_nc = pca_nc.transform(nmf_matrix_nc)
target_names_nc = uk_us.country_digit.values
pca_df = pd.DataFrame(pcafeatures_train_nc.round(5))

plot_PCA_2D_full(pcafeatures_train_nc, target=target_names_nc, target_names=['UK', 'US'], save_name='data_viz/no_char_full')

plot_PCA_2D_zoomed(pcafeatures_train_nc, target=target_names_nc, target_names=['UK', 'US'], xlim=[-0.75,0.75], ylim=[-0.75,0.75], save_name='data_viz/no_char_zoomed')
#======================================================================================================
# ANALYZE TOPICS - COSINE SIMILARITY

nmf_matrix_nc, nmf_topics_nc = nmf_on_data(vectorizer_nc, uk_us.for_topic_analysis, topics=20, num_top_words=5)
nmf_matrix_nc

Vt = pd.DataFrame(nmf_matrix_nc.round(5))
Vt.shape

# Find cosine similarity of all unique pairs of episodes. Take only those with cosine_similarity > 0.5
mt = np.array([[1,1], [1,1]])
id = np.array([[1,0], [0,1]])

similar_epiodes = []
id_episodes = []

combi = list(combinations(np.arange(0, 28), 2))
print(len(combi))
for comb in combi:
    cos_sim = cosine_similarity((Vt.iloc[comb[0]], Vt.iloc[comb[1]])).round()
    print(cos_sim)
    if np.array_equal(cos_sim, mt):
        similar_epiodes.append(comb)
    elif np.array_equal(cos_sim, id):
        id_episodes.append(comb)


sim_ranking = []
for episode in similar_epiodes:
    cos_sim = cosine_similarity((Vt.iloc[episode[0]], Vt.iloc[episode[1]])).round(2)
    sim_ranking.append((episode, cos_sim[0][1]))

sim_ranking.sort(key=lambda x: x[1], reverse=True)

ranking_df = pd.DataFrame()
ranking_df['rank_order'] = list(np.arange(0,17))
ranking_df['episode_pair'] = [x[0] for x in sim_ranking]
ranking_df['cosine_similarity'] = [x[1] for x in sim_ranking]


# Plot cosine similarities > 0.5
objects = ranking_df.rank_order.values
y_pos = np.arange(len(objects))
performance = ranking_df.cosine_similarity.values

barplot = plt.bar(y_pos, performance, align='center', alpha=1)
barplot[0].set_color('#ab1239')
barplot[1].set_color('#6a79f7')
barplot[2].set_color('#6a79f7')
barplot[3].set_color('#6a79f7')
barplot[4].set_color('#6a79f7')
barplot[5].set_color('#6a79f7')
barplot[6].set_color('#ab1239')
barplot[7].set_color('#6a79f7')
barplot[8].set_color('#6a79f7')
barplot[9].set_color('#6a79f7')
barplot[10].set_color('#6a79f7')
barplot[11].set_color('#6a79f7')
barplot[12].set_color('#ab1239')
barplot[13].set_color('#6a79f7')
barplot[14].set_color('#1fa774')
barplot[15].set_color('#1fa774')
barplot[16].set_color('#6a79f7')
plt.xticks(y_pos, objects)
plt.ylim([0.5,1])
plt.grid(False)
plt.savefig("data_viz/cos_sim_bars2.png", bbox_inches='tight')
plt.show()

#======================================================================================================
# ANALYZE TOPICS - NORMAL VS CRAZY

nmf_matrix_nc, nmf_topics_nc = nmf_on_data(vectorizer_nc, uk_us.for_topic_analysis, topics=20, num_top_words=5)

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
topic_df.normal_crazy.value_counts()
topic_df.country.value_counts()

topic_df.groupby(['country', 'normal_crazy']).size()

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

my_pal = {"UK": "#6a79f7", "US": "#1fa774"}
sns.catplot(x="normal_crazy", y="topic_mean", hue="episode_country", data=topic_df_all,
            kind="bar", palette=my_pal, height=6, ci=None).savefig("data_viz/topic_bars3.png", bbox_inches='tight')

            .savefig("data_viz/topic_bars3.png", bbox_inches='tight')



#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================
# TRIED BUT DID NOT WORK AS WELL/MAKE SENSE FOR THIS PROJECT

# GENSIM, word embeddings

tokenized_docs = [gensim.utils.simple_preprocess(d) for d in us_transcipts]
mydict = gensim.corpora.Dictionary()
mycorpus = [mydict.doc2bow(doc, allow_update=True) for doc in tokenized_docs]
doc_term_matrix = gensim.matutils.corpus2dense(mycorpus, num_terms=len(mydict))

tfidf = gensim.models.TfidfModel(mycorpus)
tfidf_matrix = gensim.matutils.corpus2dense(tfidf[mycorpus], num_terms=len(mydict))

# model = gensim.models.Word2Vec(tokenized_docs, size=10, window=2,min_count=1, sg=1)
# print(model['human'])
# What to do with these word embeddings? Transfer learning with neural network
google_vec_file = '/Users/Yvonne/Downloads/GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(google_vec_file, binary=True)
model['michael']
model.most_similar(positive=['Paris', 'England'], negative=['London'])

#======================================================================================================
# CLUSTERING
def display_cluster(X,km=[],num_clusters=0):
    color = 'brgcmyk'
    alpha = 0.5
    s = 20
    if num_clusters == 0:
        plt.scatter(X[:,0],X[:,1],c = color[0],alpha = alpha,s = s)
    else:
        for i in range(num_clusters):
            plt.scatter(X[km.labels_==i,0],X[km.labels_==i,1],c = color[i],alpha = alpha,s=s)
            plt.scatter(km.cluster_centers_[i][0],km.cluster_centers_[i][1],c = color[i], marker = 'x', s = 100)

def km_and_plot_clusters(data, num_clusters, n_init):
    km = KMeans(n_clusters=num_clusters,random_state=10,n_init=n_init) # n_init, number of times the K-mean algorithm will run
    km.fit(data)
    return display_cluster(us_lsa,km,num_clusters)


us_lsa, us_lsa_topics = lsa_on_data(vectorizer, us_transcipts, topics=8, num_top_words=5)
us_nmf, us_nmf_topics = nmf_on_data(vectorizer, us_transcipts, topics=8, num_top_words=5)

# Goes through each document and finds amount of each topic in the document
# Try putting US and UK together and see if they can separate
us_lsa
us_nmf

# Clusters are representative in topic space - each cluster could be episodes that have a high proportion of topic 1,2, etc
km_and_plot_clusters(us_lsa, num_clusters=6, n_init=3)
#
km_and_plot_clusters(us_nmf, num_clusters=6, n_init=3)


num_clusters_list = range(1,11)

inertia_list = []
for num_clusters in num_clusters_list:
    km = KMeans(n_clusters=num_clusters,random_state=10,n_init=1) # n_init, number of times the K-mean algorithm will run
    km.fit(us_nmf)
    inertia_list.append(km.inertia_)

sns.lineplot(y=inertia_list, x=num_clusters_list);
sns.lineplot(y=inertia_list, x=num_clusters_list);
