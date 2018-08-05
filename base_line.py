
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time
t1=time.time()


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline


# with open('/home/tom/new_data/train_set.txt') as file:
#     lines=file.readlines()
#
# dataset=[]
# label=[]
# type_='train'
# for line in lines:
#     if type_ != 'test':
#         dataset.append(line.split('__label__')[0])
#         label.append(line.split('__label__')[1])
# vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
# trn_term_doc = vec.fit_transform(dataset)
# Benchmark classifiers



column = "word_seg"
train = pd.read_csv('../new_data/train_set.csv')
test = pd.read_csv('../new_data/test_set.csv')
test_id = test["id"].copy()
vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
X_train = vec.fit_transform(train[column])
X_test = vec.transform(test[column])
fid0=open('baseline.csv','w')

y_train=(train["class"]-1).astype(int)

X_train, X_train_test, y_train, y_train_test = train_test_split(X_train,
                                                            y_train,
                                                            test_size=0.1)


def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time.time()
    #
    pred = clf.predict(X_test)
    test_time = time.time() - t0
    print("test time:  %0.3fs" % test_time)

    train_pred = clf.predict(X_train_test)
    score = metrics.accuracy_score(y_train_test, train_pred)
    print("accuracy:   %0.3f" % score)
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time



n_estimator=100




###gbdt
grd = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc = OneHotEncoder()
grd_lm = LogisticRegression()
grd.fit(X_train, y_train)
y_pred_grd = grd.predict_proba(X_test)[:, 1]
fpr_grd, tpr_grd, _ = roc_curve(y_train_test, y_pred_grd)
print('gbdt --{}---{}'.format(fpr_grd,tpr_grd))

###gbdt+lr
grd_enc.fit(grd.apply(X_train)[:, :, 0])
grd_lm.fit(grd_enc.transform(grd.apply(X_train_test)[:, :, 0]), y_train_test)
y_pred_grd_lm = grd_lm.predict_proba(
    grd_enc.transform(grd.apply(X_train_test)[:, :, 0]))[:, 1]
fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_train_test, y_pred_grd_lm)
print('gbdt+lr --{}---{}'.format(fpr_rt_lm,tpr_rt_lm))

###svm
lin_clf = svm.LinearSVC()
lin_clf.fit(X_train,y_train)
preds = lin_clf.predict(X_test)
svm_pred = lin_clf.predict(X_train_test)
fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_train_test, svm_pred)
print('svm --{}---{}'.format(fpr_rt_lm,tpr_rt_lm))



i=0
fid0.write("id,class"+"\n")
for item in preds:
    fid0.write(str(i)+","+str(item+1)+"\n")
    i=i+1
fid0.close()

