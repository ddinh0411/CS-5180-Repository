#-------------------------------------------------------------
# AUTHOR: Daniel Dinh
# FILENAME: seach_engine.py
# SPECIFICATION: Program to take in documents and rank them in order of relevancy to query "I love dogs"
# FOR: CS 5180- Assignment #1
# TIME SPENT: 3-4 hours total
#-----------------------------------------------------------*/

# ---------------------------------------------------------
#Importing some Python libraries
# ---------------------------------------------------------
import csv
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer

documents = []

# ---------------------------------------------------------
# Reading the data in a csv file
# ---------------------------------------------------------
with open('collection.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
         if i > 0:  # skipping the header
            documents.append (row[0])

# ---------------------------------------------------------
# Print original documents
# ---------------------------------------------------------
# --> add your Python code here

print("Documents: ", documents)

# ---------------------------------------------------------
# Instantiate CountVectorizer informing 'word' as the analyzer, Porter stemmer as the tokenizer, stop_words as the identified stop words,
# unigrams and bigrams as the ngram_range, and binary representation as the weighting scheme
# ---------------------------------------------------------
# --> add your Python code here

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS #Import in sklearns list of english stop words

# Custom Tokenizer Object/Class
class StemTokenizer:
    def __init__(self):
        self.stemmer = PorterStemmer()
    def __call__(self, doc): #Function for Tokenizer
        tokens_list = doc.lower().split() # Normalizes text to all be lowercase and split by tokens
        terms = []
        for token in tokens_list:
            token = token.strip(".,?!") # Removes Punctuation from Token
            if token not in ENGLISH_STOP_WORDS: # Removes Tokens from list if they appear in English Stop Words
                term = self.stemmer.stem(token) # Stemming all Token -> Terms
                terms.append(term) # Adds to Term List
        return terms

vectorizer = CountVectorizer(tokenizer= StemTokenizer(), token_pattern= None, binary=True, ngram_range=(1,2)) 

# ---------------------------------------------------------
# Fit the vectorizer to the documents and encode the them
# ---------------------------------------------------------
# --> add your Python code here

vectorizer.fit(documents) # Fits over Documents
document_matrix = vectorizer.transform(documents) 

# ---------------------------------------------------------
# Inspect vocabulary
# ---------------------------------------------------------
print("Vocabulary:", vectorizer.get_feature_names_out().tolist())

# ---------------------------------------------------------
# Fit the vectorizer to the query and encode it
# ---------------------------------------------------------
# --> add your Python code here

query = ["I love dogs"]
query_vector = vectorizer.transform(query) # Provides query to model

# ---------------------------------------------------------
# Convert matrices to plain Python lists
# ---------------------------------------------------------
# --> add your Python code here

doc_vectors = document_matrix.toarray().tolist()
query_vector = query_vector.toarray()[0]

# ---------------------------------------------------------
# Compute dot product
# ---------------------------------------------------------

scores = []
for vector in doc_vectors:
    score = sum(q_val * d_val for q_val, d_val in zip(query_vector, vector))
    scores.append(int(score))

print("Document Scores: ", scores)

# ---------------------------------------------------------
# Sort documents by score (descending)
# ---------------------------------------------------------

ranks = sorted(enumerate(scores), key=lambda x:x[1], reverse=True)
print("Ranking {Document Number, Relevancy Score}: ", ranks)
