
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time
t1=time.time()
#1

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
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
# from xgboost.sklearn import XGBClassifier

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

import pandas as pd
import numpy as np
column = "word_seg"
train = pd.read_csv('../input_data/train.csv')
test = pd.read_csv('../input_data/test.csv')
test_id = test["id"].copy()
vec = TfidfVectorizer(ngram_range=(1,2),min_df=0.01, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
X_train = vec.fit_transform(train[column])
X_test = vec.transform(test[column])
fid0=open('baseline.csv','w')

y_train=(train["class"]-1).astype(int)

X_train, X_train_test, y_train, y_train_test = train_test_split(X_train,
                                                            y_train,
                                                            test_size=0.1)

print('load data complete {}'.format(X_train.shape))

lin_clf = svm.LinearSVC()
lin_clf.fit(X_train,y_train)
preds = lin_clf.predict(X_test)
svm_pred = lin_clf.predict(X_train_test)

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
    print('X_train_test')
    train_pred = clf.predict(X_train_test)
    print(classification_report(y_train_test, train_pred))

    score = metrics.accuracy_score(y_train_test, train_pred)
    print("accuracy:   {}" .format(score) )

    print('X_train')
    train_pred = clf.predict(X_train)
    print(classification_report(y_train, train_pred))
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time,pred

results=[]

from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier( estimators=[('svmsvc', svm.SVC(probability=True)), ("lr", LogisticRegression(C=4, dual=True)),("rf", RandomForestClassifier(n_estimators=100)), ("svc", MultinomialNB())],voting="soft" )

test_pred={}
test_pred['id']=test_id
for clf, name in (
        (voting_clf, 'voting'),
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        # # (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        # # (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (MultinomialNB(),'MultinomialNB'),
        (RandomForestClassifier(n_estimators=500,n_jobs=-1), "Random forest"),
        ( svm.SVC(decision_function_shape='ovo'),'svmsvc'),
        (LogisticRegression(C=4, dual=True,n_jobs=4),"Logistic Regression"),
            (svm.LinearSVC(),"LinearSVC")
):
    print('=' * 80)
    print(name)
    clf_descr, score, train_time, test_time, pred=benchmark(clf)
    test_pred[name]=pred
    results.append((clf_descr, score, train_time, test_time))

test_pred_pd=pd.DataFrame.from_dict(test_pred)
test_pred_pd.to_csv('../output/base.csv',index=False,index_label=False,header=True)
print(results)
n_estimator=100


# 投票分类器

# voting_clf.fit( X_train, y_train )



#
#
# ###svm
# print('svm ')

#
# print(' svm precsion and  recall  and  f1 score as follow:\n')
# print(classification_report(y_train_test, svm_pred))
# svm_pred = lin_clf.predict(X_train)
# print(' svm train as follow:\n')
# print(classification_report(y_train, svm_pred))
#
#
# #lr
# print('lr')
# clf = LogisticRegression(C=4, dual=True)
# clf.fit(X_train, y_train)
# lr_preds=clf.predict_proba(X_train_test)
# lr_preds=np.argmax(lr_preds,axis=1)
# print(' lr precsion and  recall  and  f1 score as follow:\n')
# print(classification_report(y_train_test, lr_preds))
#
#
# ###gbdt
# print('gbdt')
# grd = GradientBoostingClassifier(n_estimators=n_estimator)
# grd_enc = OneHotEncoder()
# grd_lm = LogisticRegression()
# grd.fit(X_train, y_train)
# y_pred_grd = grd.predict_proba(X_test)[:, 1]
# print(classification_report(y_train_test, y_pred_grd))
#
#
# ###gbdt+lr
#
# print('gbdt +lr ')
# grd_enc.fit(grd.apply(X_train)[:, :, 0])
# grd_lm.fit(grd_enc.transform(grd.apply(X_train)[:, :, 0]), y_train)
#
#
# y_pred_grd_lm = grd_lm.predict_proba(
#     grd_enc.transform(grd.apply(X_train_test)[:, :, 0]))[:, 1]
#
#
# print(classification_report(y_train_test, y_pred_grd_lm))



#
# #生成提交结果
# preds=np.argmax(preds,axis=1)
# test_pred=pd.DataFrame(preds)
# test_pred.columns=["class"]
# test_pred["class"]=(test_pred["class"]+1).astype(int)
# print(test_pred.shape)
# print(test_id.shape)
# test_pred["id"]=list(test_id["id"])
# test_pred[["id","class"]].to_csv('../sub/sub_lr_baseline.csv',index=None)
# t2=time.time()
# print("time use:",t2-t1)


# #xgboost
# silent=0 ,#设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
# #nthread=4,# cpu 线程数 默认最大
# learning_rate= 0.3, # 如同学习率
# min_child_weight=1,
# # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
# #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
# #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
# max_depth=6, # 构建树的深度，越大越容易过拟合
# gamma=0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
# subsample=1, # 随机采样训练样本 训练实例的子采样比
# max_delta_step=0,#最大增量步长，我们允许每个树的权重估计。
# colsample_bytree=1, # 生成树时进行的列采样
# reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
# #reg_alpha=0, # L1 正则项参数
# #scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
# objective= 'multi:softmax', #多分类的问题 指定学习任务和相应的学习目标
# #num_class=10, # 类别数，多分类与 multisoftmax 并用
# n_estimators=100, #树的个数
# seed=1000 #随机种子
# #eval_metric= 'auc'
# )
# clf.fit(X_train,y_train,eval_metric='auc')
# y_true, y_pred = y_test, clf.predict(X_test)
# print"Accuracy : %.4g" % metrics.accuracy_score(y_true, y_pred)




# i=0
# fid0.write("id,class"+"\n")
# for item in preds:
#     fid0.write(str(i)+","+str(item+1)+"\n")
#     i=i+1
# fid0.close()

