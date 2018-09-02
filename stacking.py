from heamy.dataset import Dataset
from heamy.estimator import  Classifier
from heamy.pipeline import ModelsPipeline
from sklearn import cross_validation

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from xgboost.sklearn import XGBClassifier
from sklearn import svm


from sklearn.metrics import f1_score
#加载数据集
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import tarfile
import codecs

archive_path='/home/tom/scikit_learn_data/20news_home/20news-bydate.tar.gz'
target_dir='/home/tom/scikit_learn_data/20news_home'


# tarfile.open(archive_path, "r:gz").extractall(path=target_dir)1

data = fetch_20newsgroups()
X, y = data['data'], data['target']
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.1, random_state=111)
#创建数据集
dataset = Dataset(X_train,y_train,X_test)
#创建RF模型和LR模型
model_nb = Classifier(dataset=dataset, estimator=MultinomialNB,parameters={'voting':"soft",'n_jobs':-1},name='nb')
model_lr = Classifier(dataset=dataset, estimator=LogisticRegression, parameters={'C':4, 'dual':True,'n_jobs':-1},name='lr')
model_svm = Classifier(dataset=dataset, estimator=svm.SVC, parameters={ 'probability':True},name='svm')
model_per = Classifier(dataset=dataset, estimator=Perceptron, parameters={ 'n_iter':50,'penalty':'l2','n_jobs':-1},name='Perceptron')
# Stack两个模型mhg
# Returns new dataset with out-of-fold prediction
pipeline = ModelsPipeline(model_nb,model_lr,model_svm,model_per)
stack_ds = pipeline.stack(k=10,seed=111)
#第二层使用lr模型stack
stacker = Classifier(dataset=stack_ds, estimator=LogisticRegression)
results = stacker.predict()
# 使用10折交叉验证结果
results10 = stacker.validate(k=10,scorer=f1_score)

