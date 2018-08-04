# _*_coding:utf-8 _*_
import fasttext
#author linxinzhu
#load训练好的模型
classifier = fasttext.load_model('lab3fenci.model.bin', label_prefix='__label__')

i=0
f = open("evaluation_public.tsv", 'r')
outf = open("sub.csv",'w')
for line in f:
    outline=""
    if i==400000:
         break
    r = ""
    try:
        r = line.decode("UTF-8")
    except:
        print "charactor code error UTF-8"
        pass
    if r == "":
        try:
            r = line.decode("GBK")
        except:
            print "charactor code error GBK"
            pass
    line=line.strip()
    l_ar=line.split("\t")
    id=l_ar[0]
    title=l_ar[1]
    content=l_ar[2]
    s=""
    li=[]
    li=list()
    s="".join(content)
    li=s.split("$$$$$$$$")
    texts=li
    labels = classifier.predict(li)
    #print id
    #print labels
    strlabel=str(labels)
    if strlabel=="POSITIVE" :
        outline = id+"," + "POSITIVE" + "\n"
        outf.write(outline)
    if strlabel=="NEGATIVE" :
        outline = id+"," + "NEGATIVE" + "\n"
        outf.write(outline)
    i=i+1
    del s
    del li
f.close()
outf.close()