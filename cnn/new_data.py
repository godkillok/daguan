import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
column = "word_seg"
train = pd.read_csv('C:/work/input_data/train.csv')
vec = TfidfVectorizer(ngram_range=(1,2),min_df=0, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(train[column])
print('gg')