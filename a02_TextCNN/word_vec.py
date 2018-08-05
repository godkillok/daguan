
import word2vec

PATH='''C:/work/text_classification-master/'''



outf = open(PATH+'out.txt','w')
with open(PATH+'demo.txt','r',encoding='UTF-8') as zhihu_f:
    lines = zhihu_f.readlines()
    result=[]
    for l in lines:
        outf.write(l.split('\t')[0]+'\n')

word2vec.word2vec('out.txt',
                  PATH+'out.bin', size=200, verbose=True)

