# _*_coding:utf-8 _*_

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer


type_='train'
dataset_path='../input_data/{}.csv'.format(type_)
out_put='../input_data/{}_tfidf_more.csv'.format(type_)
with open(dataset_path) as file:
    lines=file.readlines()

step=65000

with open(out_put,'w') as f:
    for i in range(0,len(lines),step):
        dataset = []
        label=[]
        count = 0
        for line in lines:

            if i<=count<i+step:
                if type_!='test':
                    dataset.append(line.split(',')[1].split(' '))
                    label.append(line.split(',')[2])
                else:
                    dataset.append(line.split(',')[1].split(' '))
                    label.append(line.split(',')[0])
            count += 1


        from gensim.models import TfidfModel
        from gensim.corpora import Dictionary

        dct = Dictionary(dataset)
        corpus = [dct.doc2bow(line) for line in dataset]  # convert corpus to BoW format
        model = TfidfModel(corpus)  # fit model
        # vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)

        id2token={}
        for (k,v) in dct.token2id.items():
            id2token[v]=k

        # ver_rs=[]
        # vs=['520477','816903','995362','920327','1226448','1025743','990423',
        #                              '133940','1071452','876555','323159','572782','105283','166959',
        #                              '235896','554251','','1267351','1224594','201789','824446','263278']
        # for v in vs:
        #     print(v,dct.token2id[v])
        dataset_after_tfidf=[]
        for i in range(len(dataset)):
            vector = model[corpus[i]]  # apply model to the first corpus document
            ver_rs=[]
            for  v in vector:
                (id2,score)=v
                if score>0.01:
                    ver_rs.append(id2token[id2])
            d_temp=[]
            for d in dataset[i]:
                if type_ != 'test':
                    if d in ver_rs or d=='816903':
                       d_temp.append(d)
                else:
                    if d in ver_rs:
                        d_temp.append(d)
            # print(len(dataset[i]),len(d_temp))
            dataset_after_tfidf.append(' '.join(d_temp))




        if type_!='test':
            # f.writelines(dataset_after_tfidf[i]+'__label__'+label[i]+'\n')
            for i in range(len(dataset_after_tfidf)):
                count = 0
                new_line = []
                for t in dataset_after_tfidf[i].split(' '):
                    if t != '816903' or count < 15:
                        if t == '816903':
                            count += 1
                        if t not in ['520477', '816903', '995362', '920327', '1226448', '1025743', '990423',
                                     '133940', '1071452', '876555', '323159', '572782', '105283', '166959',
                                     '235896', '554251', '', '1267351', '1224594', '201789', '824446', '263278']:
                            new_line.append(t)
                    else:
                        # print(count)
                        count = 0
                        file.writelines(' '.join(new_line) + ' __label__' + str(label[i]) + '\n')
                        new_line = []
            else:
                f.writelines(label[i]+','+dataset_after_tfidf[i]  + '\n')

        print('dd')

#
#
# vectorizer = TfidfVectorizer()
# vectorizer.fit_transform(dataset)
# a = vectorizer.fit_transform(dataset)
# print(a[0])
# print(vectorizer.get_feature_names())
#
