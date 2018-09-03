import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
#12813123
column = "word_seg"
new_ = pd.read_csv('./cnn/valid_id')
train = pd.read_csv('../input_data/train.csv')
test = pd.read_csv('../input_data/test.csv')


new_=pd.merge(new_, test, how='inner', on=['id', 'id'])
print('merge_before')
print(train._info_axis)
print(train.shape)
train = train.append(new_)
train.to_csv('../input_data/new_train.csv',index=False,index_label=False,header=True)
print('merge_after')
print(train._info_axis)
print(train.shape)
#
#
test_id = test["id"].copy()
# vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
# trn_term_doc = vec.fit_transform(train[column])
# test_term_doc = vec.transform(test[column])
# fid0=open('baseline.csv','w')
#
# y=(train["class"]-1).astype(int)
# lin_clf = svm.LinearSVC()
# lin_clf.fit(trn_term_doc,y)
#
# preds = lin_clf.predict(test_term_doc)
#
# i=0
# fid0.write("id,class"+"\n")
# for item in preds:
#     fid0.write(str(i)+","+str(item+1)+"\n")
#     i=i+1
# fid0.close()