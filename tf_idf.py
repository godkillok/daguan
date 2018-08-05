# _*_coding:utf-8 _*_

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

X_train = ['This is the first document.', 'This is the second document.']
X_test = ['This is the third document.']
vectorizer = TfidfVectorizer()
# 用X_train数据来fit
vectorizer.fit(X_train)
# 得到tfidf的矩阵
tfidf_train = vectorizer.transform(X_train)
tfidf_test = vectorizer.transform(X_test)

tfidf_train.toarray()
print(vectorizer.get_feature_names())
# vectorizer = TfidfVectorizer()
# vectorizer.fit_transform(dataset)
# a = vectorizer.fit_transform(dataset)
# print(a[0])
# print(vectorizer.get_feature_names())