# _*_coding:utf-8 _*_

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer


import pandas as pd
import numpy as np
column = "word_seg"
train_set_path='../input_data/train_tfidf_more.txt'
test_set_path='../input_data/test_tfidf_more.txt'
def  prepare_data_set():

    train = pd.read_csv('../input_data/train.csv')
    test = pd.read_csv('../input_data/test.csv')
    test_id = test["id"].copy()
    vec = TfidfVectorizer(ngram_range=(1,1),min_df=0.01, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
    x_train=train[column]
    x_test=test[column]
    y_train=(train["class"])

    vec.fit(x_train)
    feature_name=vec.get_feature_names()
    x_train_clear=[]


    for tr in x_train:
        x_train_clear.append(' '.join([t for t in tr.split(' ') if t  in feature_name]) )

    x_test_clear=[]
    for tr in x_test:
        x_test_clear.append(' '.join([t for t in tr.split(' ') if t  in feature_name]) )




    with open(train_set_path,'w') as f:
        for t in range(len(x_train_clear)):
            f.writelines(x_train_clear[t]+' __label__'+y_train[t]+'\n')


    with open(test_set_path,'w') as f:
        for t in range(len(x_test_clear)):
            f.writelines(x_test_clear[t]+'\n')

def pred():
    import  fastText
    classifier = fastText.load_model('../input/sur_tfidf_more.bin')
    train_result = classifier.test(train_set_path)
    print(train_result)

    with open(test_set_path) as file:
        lines = file.readlines()
    result_id = []
    result_label2 = []

    for line in lines:
        if len(line) < 2:
            continue
        (_id, text) = line.split(',')
        result_id.append(_id)
        result_label2.append(classifier.predict(text.strip())[0][0].replace('__label__', ''))

    print(len(result_id))
    print(len(result_label2))
    result = {
        'id': result_id,
        'class': result_label2
    }

    result_pd = pd.DataFrame.from_dict(result)

    result_pd.to_csv('../output/result.csv',columns=['id','class'],index=False,index_label=False,header=False)

prepare_data_set()




