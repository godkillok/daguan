from gensim import corpora, models, similarities
docs = [['Looking', 'for', 'the', 'meanings', 'of', 'words'],
        ['phrases'],
        ['and', 'expressions'],
        ['We', 'provide', 'hundreds', 'of', 'thousands', 'of', 'definitions'],
        ['synonyms'],
        ['antonyms'],
        ['and', 'pronunciations', 'for', 'English', 'and', 'other', 'languages'],
        ['derived', 'from', 'our', 'language', 'research', 'and', 'expert', 'analysis'],
        ['We', 'also', 'offer', 'a', 'unique', 'set', 'of', 'examples', 'of', 'real', 'usage'],
        ['as', 'well', 'as', 'guides', 'to:']]
dictionary = corpora.Dictionary(docs)
corpus = [dictionary.doc2bow(text) for text in docs]
nf=len(dictionary.dfs)
nf=10001
corpus[0].append((10000,2))
index = similarities.SparseMatrixSimilarity(corpus, num_features=nf)
phrases = [['Looking', 'for', 'the', 'meanings', 'of', 'words']]
phrase2word=[dictionary.doc2bow(text) for text in phrases]
sims=index[phrase2word]
sims2=index[[corpus[0]]]
print(sims)
print(sims2)