
import numpy as np
import pandas as pd
from re import sub
from gensim.utils import simple_preprocess

#  this data was gathered from Kaggle database
data = pd.read_csv('data.csv', encoding='iso-8859-1')
documents  = list(data["headlines"])
query_strings = list(data["text"])


stopwords = ['the', 'and', 'are', 'a']
def preprocess(doc):
    # Tokenize, clean up input document string
    doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
    doc = sub(r'<[^<>]+(>|$)', " ", doc)
    doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
    doc = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
    return [token for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if token not in stopwords]

# Preprocess the documents, including the query string
corpus = [preprocess(document) for document in documents]
queries = [preprocess(query_string) for query_string in query_strings] 


# create a similarity matrix, that contains the similarity between each pair of words
import gensim.downloader as api
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import SoftCosineSimilarity

# for better sentiment analysis
glove = api.load("glove-wiki-gigaword-50")    
similarity_index = WordEmbeddingSimilarityIndex(glove)

# build the term dictionary, TF-idf model
dictionary = Dictionary(corpus+queries)
tfidf = TfidfModel(dictionary=dictionary)

# create the term similarity matrix.  
similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf)

# create a new dataframe
df_gensim=data.copy()

# Finally, the soft cosine similarity between the text and each of the headlines is calculated
for query in queries:
    query_tf = tfidf[dictionary.doc2bow(query)]

    index = SoftCosineSimilarity(
            tfidf[[dictionary.doc2bow(document) for document in corpus]],
            similarity_matrix)

    doc_similarity_scores = index[query_tf]

    # output the sorted similarity headline and text
    sorted_indexes = np.argsort(doc_similarity_scores)[::-1]
    for headline_index in sorted_indexes:
        df_gensim.loc[i,"matching rank of "+ str(headline_index) ]= df_gensim.loc[headline_index,"original_headline"]
