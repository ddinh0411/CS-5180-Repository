#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 5180- Assignment #4
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

# importing required libraries
import pandas as pd

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------------------------------------------------------
# Helper function: tokenize text and remove stopwords only
# ---------------------------------------------------------
def preprocess(text):
    # --> add your Python code here
    # Suggested steps:
    # 1. convert text to lowercase
    # 2. split into tokens
    # 3. remove stopwords only
    # 4. return the filtered tokens
    pass


# ---------------------------------------------------------
# 1. Load the input files
# ---------------------------------------------------------
# Files:
#   docs.csv
#   queries.csv
#   relevance_judgments.csv
# --> add your Python code here


# ---------------------------------------------------------
# 2. Build the BM25 index for the documents
# ---------------------------------------------------------
# Requirement: remove stopwords only
# Steps:
#   1. preprocess each document
#   2. store tokenized documents in a list
#   3. create the BM25 model
# --> add your Python code here


# ---------------------------------------------------------
# 3. Process each query and compute AP values
# ---------------------------------------------------------
# Suggested structure:
#   - for each query:
#       1. preprocess the query
#       2. compute BM25 scores for all documents
#       3. rank documents by score in descending order
#       4. retrieve the relevant documents for that query
#       5. compute AP
# --> add your Python code here


    # -----------------------------------------------------
    # 4. Compute Average Precision (AP)
    # -----------------------------------------------------
    # Suggested steps:
    #   - initialize variables
    #   - go through the ranked documents
    #   - whenever a relevant document is found:
    #         precision = (# relevant found so far) / (current rank position)
    #         add precision to the running sum
    #   - AP = sum of precisions / total number of relevant documents
    #   - if there are no relevant documents, AP = 0

    # store the AP value for this query (use any data structure you prefer)


# ---------------------------------------------------------
# 5. Sort queries by AP in descending order
# ---------------------------------------------------------
# --> add your Python code here


# ---------------------------------------------------------
# 6. Print the sorted queries and their AP scores
# ---------------------------------------------------------
print("====================================================")
print("Queries sorted by Average Precision (AP):")
# --> add your Python code here