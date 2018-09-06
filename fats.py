# _*_coding:utf-8 _*_

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import pandas as pd
import fastText as f
def create_data():
    type_='test'
    dataset_path='../input_data/{}.csv'.format(type_)
    out_put='../input_data/{}_more_data.csv'.format(type_)
    with open(dataset_path) as file:
        lines=file.readlines()

    step=65000

    with open(out_put,'w') as file:
        dataset=[]
        for line in lines:
            if 'id' in line:
                continue
            line_list=line.strip().split(',')[1].split(' ')
            new_line=[]

            for t in line_list:
                if t not in ['520477', '816903', '995362', '920327', '1226448', '1025743', '990423',
                             '133940', '1071452', '876555', '323159', '572782', '105283', '166959',
                             '235896', '554251', '', '1267351', '1224594', '201789', '824446', '263278']:
                    new_line.append(t)
            label=line.split(',')[0]
            file.writelines(' '.join(new_line) + ' __label__' + str(label) + '\n')
def pred_():
    import numpy as np
    model=f.load_model('../new_more_data_n51.bin')
    type_ = 'test'
    dataset_path = '../input_data/{}.csv'.format(type_)
    out_put = '../input_data/{}_more_data_sc.csv'.format(type_)
    with open(dataset_path) as file:
        lines=file.readlines()
    id_list = []
    label_list=[]
    for  l in lines:
        if 'id' in l:
            lines.remove(l)
            break

    score_array=np.zeros((len(lines),19))
    for li,line in enumerate(lines):
        line_list = line.strip().split(',')[1].split(' ')
        new_line = []

        for t in line_list:
            if t not in ['520477', '816903', '995362', '920327', '1226448', '1025743', '990423',
                         '133940', '1071452', '876555', '323159', '572782', '105283', '166959',
                         '235896', '554251', '', '1267351', '1224594', '201789', '824446', '263278']:
                new_line.append(t)
        id = line.split(',')[0]
        id_list.append(id)
        lab=model.predict(' '.join(new_line),20)

        for i in range(len(lab[0])):
            idx=int(lab[0][i].replace('__label__', ''))-1
            sc=lab[1][i]
            score_array[li,idx]=round(sc,4)

        label_list.append(int(lab[0][0].replace('__label__','')))

    score_pd = pd.DataFrame(score_array)
    score_pd.to_csv(out_put,index_label=None,index=False)
    label_pd=pd.DataFrame.from_dict({
        'id':id_list,
        'class':label_list
    }
                                    )
    out_put = '../fast_n51.csv'
    label_pd[['id','class']].to_csv(out_put, index_label=None, index=False)
pred_()
