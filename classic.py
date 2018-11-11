# coding=gbk



import time

from sklearn import metrics

import pickle as pickle
from sklearn.ensemble import VotingClassifier
import pandas as pd
import random
import pickle

folder_num = 10
label_num = 7
import numpy as np


# Multinomial Naive Bayes Classifier  1

def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB

    model = MultinomialNB(alpha=0.01)

    model.fit(train_x, train_y)

    return model


# KNN Classifier  

def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier

    model = KNeighborsClassifier()

    model.fit(train_x, train_y)

    return model


# Logistic Regression Classifier  

def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(penalty='l2')

    model.fit(train_x, train_y)

    return model


# Random Forest Classifier  

def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=8)

    model.fit(train_x, train_y)

    return model


# Decision Tree Classifier  

def decision_tree_classifier(train_x, train_y):
    from sklearn import tree

    model = tree.DecisionTreeClassifier()

    model.fit(train_x, train_y)

    return model


# GBDT(Gradient Boosting Decision Tree) Classifier  

def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier

    model = GradientBoostingClassifier(n_estimators=200)

    model.fit(train_x, train_y)

    return model


# SVM Classifier  

def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC

    model = SVC(kernel='rbf', probability=True)

    model.fit(train_x, train_y)

    return model


# SVM Classifier using cross validation  

def svm_cross_validation(train_x, train_y):
    from sklearn.grid_search import GridSearchCV

    from sklearn.svm import SVC

    model = SVC(kernel='rbf', probability=True)

    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}

    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)

    grid_search.fit(train_x, train_y)

    best_parameters = grid_search.best_estimator_.get_params()

    for para, val in list(best_parameters.items()):
        print(para, val)

    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)

    model.fit(train_x, train_y)

    return model


def train_one_label():
    model_saver={} #first level is  label_num,second level is classifier, third one is folder
    for la in range(label_num):
        model_save[la]=train_classifiers(la)


def read_data(mo, label_i, data_file='data.pkl'):
    with open(data_file, 'rb') as f:
        model_data = pickle.dump(f)

    samples = model_data[label_i][mo]
    train = samples[:int(len(samples) * 0.9)]
    test = samples[int(len(samples) * 0.9):]

    train_x = [sa[0] for sa in train]
    train_y = [sa[1] for sa in train]

    test_x = [sa[0] for sa in test]
    test_y = [sa[1] for sa in test]

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    return train_x, train_y, test_x, test_y


def train_classifiers(label_i):
    test_classifiers = ['NB', 'LR', 'RF', 'DT', 'SVM', 'SVMCV', 'GBDT']  # 'KNN',

    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'SVMCV': svm_cross_validation,
                   'GBDT': gradient_boosting_classifier
                   }
    model_save={}

    for classifier in test_classifiers:
        model_save[classifier]={}
        for mo in range(folder_num):
            print('******************* {}***********model_num {}*********'.format(classifier, mo))
            start_time = time.time()
            train_x, train_y, test_x, test_y = read_data(mo, label_i,'data.pkl')
            model = classifiers[classifier](train_x, train_y)
            if mo==0:
                test_set_x=test_x
                test_set_y=test_y
            else:
                test_set_x=np.concatenate((test_x,test_set_x),axis=0)
                test_set_y = np.concatenate((test_y,test_set_y),axis=0)
            print('training took %fs!' % (time.time() - start_time))

            predict = model.predict(test_x)
            model_save[classifier][mo] = model

            # precision = metrics.precision_score(test_y, predict, average='macro')
            #
            # recall = metrics.recall_score(test_y, predict, average='macro')
            #
            # print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
            #
            # accuracy = metrics.accuracy_score(test_y, predict)
            #
            # print('accuracy: %.2f%%' % (100 * accuracy))

        for mo in range(folder_num):
            predict = model_save[classifier][mo].predict(test_set_x)
            precision = metrics.precision_score(test_set_y, predict, average='macro')

            recall = metrics.recall_score(test_y, predict, average='macro')

            print('%d precision: %.2f%%, recall: %.2f%%' % (mo,100 * precision, 100 * recall))

            accuracy = metrics.accuracy_score(test_y, predict)

            print('accuracy: %.2f%%' % (100 * accuracy))

    return model_save
def  predict_hard_vote():
    pass

def create_rand_sample(path):
    with  open(path, 'r') as f:
        lines = f.readlines()
    positive = []
    negative = []
    positive_user = []
    model_data = {}

    for label_i in range(label_num):
        model_data[label_i] = {}
        for li in lines:
            li = li.strip()
            user_id, content, y_label = li.split()[0], li.split()[1], li.split()[2]
            content = content.split(',')
            if y_label.split(',')[label_i] == 1:
                positive.append([content, 1])
                positive_user.append(user_id)
            else:
                negative.append([content, 0])
        pos_count = len(positive)
        neg_count = len(negative)
        # pos_neg=int(pos_count/neg_count)
        sample_count = min(pos_count, neg_count)

        for i in range(folder_num):
            nega_sample = random.sample(negative, sample_count)
            sampels = nega_sample + positive
            random.shuffle(sampels)
            model_data[label_i][i] = sampels

    with open('data.pkl', 'wb') as f:
        pickle.dump(model_data, f)


if __name__ == '__main__':
    import sys

    try:
        data_file = sys.argv[1]
    except:
        data_file = "./trainCG.csv"

    thresh = 0.5

    model_save_file = None

    model_save = {}

    test_classifiers = ['NB', 'LR', 'RF', 'DT', 'SVM', 'SVMCV', 'GBDT']  # 'KNN',

    classifiers = {'NB': naive_bayes_classifier,

                   'KNN': knn_classifier,

                   'LR': logistic_regression_classifier,

                   'RF': random_forest_classifier,

                   'DT': decision_tree_classifier,

                   'SVM': svm_classifier,

                   'SVMCV': svm_cross_validation,

                   'GBDT': gradient_boosting_classifier

                   }

    print('reading training and testing data...')
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

    train_x_, train_y, test_x_, test_y = read_data(data_file)
    data_set = train_x_ + test_x_
    print(train_x_.head(10))
    print(data_set.shape)
    print(data_set.head(10))
    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_df=0.95, use_idf=1, smooth_idf=1, sublinear_tf=1)

    vec.fit_transform(train_x_)
    train_x = vec.transform(train_x_)
    test_x = vec.transform(test_x_)

    for classifier in test_classifiers:

        print('******************* %s ********************' % classifier)

        start_time = time.time()

        model = classifiers[classifier](train_x, train_y)

        print('training took %fs!' % (time.time() - start_time))

        predict = model.predict(test_x)

        if model_save_file != None:
            model_save[classifier] = model

        precision = metrics.precision_score(test_y, predict, average='macro')

        recall = metrics.recall_score(test_y, predict, average='macro')

        print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))

        accuracy = metrics.accuracy_score(test_y, predict)

        print('accuracy: %.2f%%' % (100 * accuracy))

    if model_save_file != None:
        pickle.dump(model_save, open(model_save_file, 'wb'))
