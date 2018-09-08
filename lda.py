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

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
# train = pd.read_csv('../input_data/train.csv')
# test = pd.read_csv('../input_data/test.csv')


with  open('../input_data/train.csv','r') as f:
    lines=f.readlines()
print(lines[1])
train_documents=[d.split(',')[1].split() for d in lines[1:100]]
labelss=[int(d.split(',')[2])-1 for d in lines[1:100]]
logging.info('documents {}'.format(train_documents[0]))
logging.info('lables {}'.format(labelss[0]))



with  open('../input_data/test.csv','r') as f:
    lines=f.readlines()
print(lines[1])
test_documents=[d.split(',')[1].split() for d in lines[1:100]]
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
for x in range(len(train_documents)):
    a1 = dictionary.doc2bow(train_documents[x])
    train_topic1.append(lsi[a1])

train_topic=np.array(train_topic1)
test_topic1=[]
for x in range(len(test_documents)):
    a1 = dictionary.doc2bow(test_documents[x])
    test_topic1.append(lsi[a1])
test_topic=np.array(test_topic1)


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
logging.info('train_topic shape {}'.format(train_topic.shape))




X_train, X_test, Y_train, Y_test = train_test_split(train_topic, labelss, test_size=0.1)
sv = svm.LinearSVC()
sv.fit(train_topic,labelss)


sv.fit(X_train, Y_train)
sv.score(X_test, Y_test)

preds = sv.predict(test_topic)
i=0
fid0=open('lda.csv','w')
fid0.write("id,class"+"\n")
for item in preds:
    fid0.write(str(i)+","+str(item+1)+"\n")
    i=i+1
fid0.close()
