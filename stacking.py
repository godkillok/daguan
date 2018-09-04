from heamy.dataset import Dataset
from heamy.estimator import  Classifier
from heamy.pipeline import ModelsPipeline
from sklearn import cross_validation

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from xgboost.sklearn import XGBClassifier
from sklearn import svm

import scipy
from sklearn.metrics import accuracy_score,f1_score
#加载数据集
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import tarfile
import codecs
import numpy as np
import pickle
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
archive_path='/home/tom/scikit_learn_data/20news_home/20news-bydate.tar.gz'
target_dir='/home/tom/scikit_learn_data/20news_home'

column='word_seg'
# tarfile.open(archive_path, "r:gz").extractall(path=target_dir)11


train = pd.read_csv('../input_data/train.csv')
test = pd.read_csv('../input_data/test.csv')
new_ = pd.read_csv('./cnn/valid_id')


#21
# new_=pd.merge(new_, test, how='inner', on=['id', 'id'])
# print('merge_before')
# print(train._info_axis)
# print(train.shape)
# train = train.append(new_)
# print('merge_after')
# print(train._info_axis)
# print(train.shape)

y=(train["class"]-1).astype(int)
read=False
if read==False:
    vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)

    trn_term_doc = vec.fit_transform(train[column])
    print(len(vec.vocabulary))
    test_term_doc = vec.transform(test[column])
    print('write to  .....')
    with open('../input_data/trn_term_doc_12.pil','wb') as f:
        pickle.dump(trn_term_doc, f)
    with open('../input_data/test_term_doc_12.pil','wb') as f:
        pickle.dump(test_term_doc, f)
else:
    logging.info('read from .....')
    with open('../input_data/trn_term_doc.pil', 'rb') as f:
        trn_term_doc=pickle.load( f)
    with open('../input_data/test_term_doc.pil', 'rb') as f:
        test_term_doc=pickle.load( f)

X_train, X_test, y_train, y_test =train_test_split(trn_term_doc, y, test_size=0.01, random_state=111)
print('tttt')
# X_train=X_train.toarray()
# X_test=X_test.toarray()
print('to array')
#创建数据集11
dataset = Dataset(X_train,y_train,test_term_doc,use_cache=False)
#创建RF模型和LR模型1

class_use_cache=False
model_nb = Classifier(dataset=dataset, estimator=MultinomialNB,name='nb',use_cache=class_use_cache)
model_lr = Classifier(dataset=dataset, estimator=LogisticRegression, parameters={'C':4, 'dual':True,'n_jobs':-1},name='lr',use_cache=class_use_cache)
model_lr2 = Classifier(dataset=dataset, estimator=LogisticRegression, parameters={'C':4, 'multi_class':'multinomial','solver':'sag','dual':False,'n_jobs':-1},name='lr2',use_cache=class_use_cache)
model_svm = Classifier(dataset=dataset, estimator=svm.SVC, parameters={ 'probability':True},name='svm',use_cache=class_use_cache)
model_svc= Classifier(dataset=dataset, estimator=svm.LinearSVC,name='LinearSVC',use_cache=class_use_cache)
model_gbdt=Classifier(dataset=dataset, estimator=GradientBoostingClassifier,name="gdbt",parameters={ 'n_estimators':50,'subsample' : 0.6},use_cache=class_use_cache)
# Stack两个模型mhg
# Returns new dataset with out-of-fold prediction,model_svm,model_per
logging.info('stack_ds....')
pipeline = ModelsPipeline(model_lr,model_svc)
# pipeline = ModelsPipeline(model_nb),model_nb,model_lr,model_lr2
stack_ds = pipeline.stack(k=2,seed=111)
#第二层使用lr模型stack2
logging.info('second layer....')
stacker = Classifier(dataset=stack_ds, estimator=GradientBoostingClassifier,use_cache=False,probability=False)
results = stacker.predict()


# 使用10折交叉验证结果
results10 = stacker.validate(k=10,scorer=accuracy_score)
logging.info(results10)




result_list=list(results+1)

test_id=list(test[["id"]].copy())
test_id=[i  for i in  range(len(result_list))]
logging.info('len of ....')
logging.info(len(result_list))
logging.info(len(test_id))
pred_dic={'class':result_list,"id":test_id}

pd.DataFrame.from_dict(pred_dic)[["id","class"]].to_csv('../output/sub_stack_nb2.csv',index=None)

# print(accuracy_score(y_test, results))1
