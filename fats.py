# _*_coding:utf-8 _*_

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer


type_='new_train'
dataset_path='../input_data/{}.csv'.format(type_)
out_put='../input_data/{}_more_data.csv'.format(type_)
with open(dataset_path) as file:
    lines=file.readlines()

step=65000

with open(out_put,'w') as f:
    dataset=[]
    for line in lines:
        line_list=line.split(',')[1].split(' ')
        new_line=[]
        for t in line_list:
            if t not in ['520477', '816903', '995362', '920327', '1226448', '1025743', '990423',
                         '133940', '1071452', '876555', '323159', '572782', '105283', '166959',
                         '235896', '554251', '', '1267351', '1224594', '201789', '824446', '263278']:
                new_line.append(t)
        label=line.split(',')[2]
        file.writelines(' '.join(new_line) + ' __label__' + str(label) + '\n')



#
#
# vectorizer = TfidfVectorizer()
# vectorizer.fit_transform(dataset)
# a = vectorizer.fit_transform(dataset)
# print(a[0])
# print(vectorizer.get_feature_names())
#
