from heamy.dataset import Dataset
from heamy.estimator import  Classifier
from heamy.pipeline import ModelsPipeline
from sklearn import cross_validation

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from xgboost.sklearn import XGBClassifier
from sklearn import svm
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
import scipy
from sklearn.metrics import accuracy_score,f1_score
#加载数据集
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
import tarfile
import codecs
import numpy as np
import pickle
import pandas as pd
import logging
import os
import re
from gensim import models,corpora
import  fastText
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
# train = pd.read_csv('../input_data/train.csv')
# test = pd.read_csv('../input_data/test.csv')


with  open('../input_data/train.csv','r') as f:
    lines=f.readlines()
print(lines[1])
train_documents=[d.split(',')[1].split() for d in lines[1:]]
labelss=[int(d.split(',')[2])-1 for d in lines[1:]]
logging.info('documents {}'.format(train_documents[0]))
logging.info('lables {}'.format(labelss[0]))



with  open('../input_data/test.csv','r') as f:
    lines=f.readlines()
print(lines[1])
test_documents=[d.split(',')[1].split() for d in lines[1:]]
logging.info('train_documents {}'.format(len(train_documents)))

documents=train_documents+test_documents
logging.info('train_documents {}'.format(len(train_documents)))
logging.info('documents {}'.format(len(documents)))

dictionary=corpora.Dictionary(documents)
corpus=[dictionary.doc2bow(doc) for doc in documents]#generate the corpus
tf_idf=models.TfidfModel(corpus)#the constructor
corpus_tfidf=tf_idf[corpus]
topic_num=500
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=topic_num)

# 建立LSA对应的文档主题矩阵
train_topic1 = []
N = topic_num


train_flag=False

if train_flag==True:
    train_vector = np.zeros((len(test_documents),N), float)
    for x in range(len(train_documents)):
        a1 = dictionary.doc2bow(train_documents[x])
        for index, value in lsi[a1]:
            train_vector[x, index] = value




    # print(dense_vector)
    # print(lsi[a1])1

    train_topic=train_vector
    test_topic1=[]
    dense_vector = np.zeros((len(test_documents),N), float)
    for x in range(len(test_documents)):
        a1 = dictionary.doc2bow(test_documents[x])
        N = topic_num

        for index, value in lsi[a1]:
            dense_vector[x,index] = value

    test_topic=dense_vector


    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    logging.info('train_topic shape {}'.format(train_topic.shape))
    np.save('./train_topic',train_topic)
    np.save('./test_topic',test_topic)
else:
    train_topic=np.load('./train_topic')
    test_topic = np.load('./test_topic')
    model=fastText.load_model('../input_data/new_more_data.bin')

    for i in train_topic.shape[0]:
        a1 = model.get_sentence_vector(train_documents[i])
        train_topic[i, :]=np.concatenate([train_topic[i,:],a1],0)

    for i in test_topic.shape[0]:
        a1 = model.get_sentence_vector(test_documents[i])
        test_topic[i, :]=np.concatenate([test_topic[i,:],a1],0)

    logging.info('train_topic shape {}'.format(train_topic.shape))
    logging.info('test_topic shape {}'.format(test_topic.shape))



X_train, X_test, Y_train, Y_test = train_test_split(train_topic, labelss, test_size=0.1)
sv = svm.LinearSVC()
sv.fit(train_topic,labelss)


sv.fit(X_train, Y_train)
logging.info(sv.score(X_test, Y_test))

preds = sv.predict(test_topic)
i=0
fid0=open('lda.csv','w')
fid0.write("id,class"+"\n")
for item in preds:
    fid0.write(str(i)+","+str(item+1)+"\n")
    i=i+1
fid0.close()

