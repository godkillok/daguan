from gensim import corpora, models, similarities,matutils
import time
t1=time.time()
# docs = [['Looking', 'for', 'the', 'meanings', 'of', 'words'],
#         ['phrases'],
#         ['and', 'expressions'],
#         ['We', 'provide', 'hundreds', 'of', 'thousands', 'of', 'definitions'],
#         ['synonyms'],
#         ['antonyms'],
#         ['and', 'pronunciations', 'for', 'English', 'and', 'other', 'languages'],
#         ['derived', 'from', 'our', 'language', 'research', 'and', 'expert', 'analysis'],
#         ['We', 'also', 'offer', 'a', 'unique', 'set', 'of', 'examples', 'of', 'real', 'usage'],
#         ['as', 'well', 'as', 'guides', 'to:']]
# dictionary = corpora.Dictionary(docs)
# corpus = [dictionary.doc2bow(text) for text in docs]
# nf=len(dictionary.dfs)
# # nf=10001
# # corpus[0].append((10000,2))
# index = similarities.SparseMatrixSimilarity(corpus, num_features=nf)
# phrases = [['Looking', 'for', 'the', 'meanings', 'of', 'words']]
# phrase2word=[dictionary.doc2bow(text) for text in phrases]
# sims=index[phrase2word]
# sims2=index[[corpus[0]]]
# print(sims)
# print(sims2)
# t2=time.time()
# print(t2-t1)
import numpy as np
import random
id = []
id_rever={}
item = []
total_item = 0
max_=0
with open('/home/tom/Desktop/tags/word_1000.txt','r',encoding='utf8') as file:
    lines=file.readlines()


    for l in lines:
        l=l.strip()
        try:
            for i in l.split('\x01')[1:]:
              max_=max(int(i.split('\x02')[0]),max_)
            item.append([(int(i.split('\x02')[0]), int(i.split('\x02')[1])) for i in l.split('\x01')[1:]])
            id.append(l.split('\x01')[0])
            id_rever[total_item]=l.split('\x01')[0]

            total_item+=1
        except:
            pass
            # print(l)

nf = min(1500000,max_+1)
print(max_+1)
result=[]
t1=time.time()
for (_id,i) in enumerate(item):
    t1 = time.time()
    index_item=[]
    per_dict = {}
    per_id=[]
    count=0
    per_dict[count]=id_rever[_id]
    index_item.append(i)
    for j in range(100):
        count +=1

        a=random.randint(0,total_item-1)
        per_dict[count]=id_rever[a]
        index_item.append(item[a])
    dense_index_item=matutils.corpus2dense(index_item, num_terms=nf)
    from sklearn.metrics.pairwise import cosine_similarity

    sims = cosine_similarity(dense_index_item,index_item[0].reshape(1,-1))
    index = similarities.SparseMatrixSimilarity(index_item, num_features=nf)
    sims=index[[i]][0:]


    si=list(np.where(sims>0.3)[1][0:])
    result.append([(per_dict[0],per_dict[s]) for s in si])
    t2=time.time()
    print(t2-t1)
print(result)



