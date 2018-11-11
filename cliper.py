from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

tfidf_vectorizer = CountVectorizer( ngram_range=(1, 3))

new_docs = ['He watches basketball and baseball',
            'Julie likes to play basketball',
            'Jane loves to play baseball']
new_term_freq_matrix = tfidf_vectorizer.fit_transform(new_docs)
print(tfidf_vectorizer.vocabulary_)