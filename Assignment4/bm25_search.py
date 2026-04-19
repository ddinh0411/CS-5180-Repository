#-------------------------------------------------------------------------
# AUTHOR: Daniel Dinh
# FILENAME: bm25_search.py
# SPECIFICATION: Same as Assignment 3, create a model (this time a BM25 model)
# and if given documents and queries, rank the documents by relevance and then
# rank the queries using AP given the judgements.
# FOR: CS 5180- Assignment #4
# TIME SPENT: actual coding portion took 1-2 hours (very similar to Assignment 3 coding)
# Q1 - Q3 2-3 hours (purely because of all the calculations)
#-----------------------------------------------------------*/

# importing required libraries
import pandas as pd
import numpy as np

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------------------------------------------------------
# Helper function: tokenize text and remove stopwords only
# ---------------------------------------------------------
def preprocess(text):
    tokens = str(text).lower().split()
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS]

# ---------------------------------------------------------
# 1. Load the input files
# ---------------------------------------------------------

docs_collection = pd.read_csv('docs.csv', header=0)
queries = pd.read_csv('queries.csv', header=0)
judgement_list = pd.read_csv('relevance_judgments.csv', header=0)

# ---------------------------------------------------------
# 2. Build the BM25 index for the documents
# ---------------------------------------------------------

tokenized_docs = [preprocess(doc) for doc in docs_collection['text']]
bm25 = BM25Okapi(tokenized_docs)

# ---------------------------------------------------------
# 3. Process each query and compute AP values
# ---------------------------------------------------------

token_queries = [preprocess(query) for query in queries['query_text']]

bm25_results = []
document_ids = docs_collection['doc_id'].values
for i in range(len(queries)):
    query_token = token_queries[i]

    bm25_score = bm25.get_scores(query_token)
    rank_indices = np.argsort(bm25_score)[::-1]
    doc_rankings = document_ids[rank_indices]

    bm25_results.append(doc_rankings)

    # -----------------------------------------------------
    # 4. Compute Average Precision (AP)
    # -----------------------------------------------------

ap_scores = []
query_ids = queries['query_id'].values

for q_id, ranked_docs in enumerate(bm25_results):
    current_query = query_ids[q_id]

    query_judgements = judgement_list[judgement_list['query_id'] == current_query]

    relevant_for_query = query_judgements[query_judgements['judgment']=='R']
    total_relevant = len(relevant_for_query)

    if total_relevant == 0:
        ap_scores.append(0.0)
        continue

    hits = 0
    sum_precision = 0.0

    for rank, doc_id in enumerate(ranked_docs, start=1):
        doc_judgement = query_judgements[query_judgements['doc_id'] == doc_id]['judgment'].values

        if len(doc_judgement) > 0 and doc_judgement[0] == 'R':
            hits += 1
            precision_at_k = hits / rank
            sum_precision += precision_at_k
        
    ap = sum_precision / total_relevant
    ap_scores.append(ap)

# ---------------------------------------------------------
# 5. Sort queries by AP in descending order
# ---------------------------------------------------------

final_ranking = sorted(zip(query_ids, ap_scores), key=lambda x: x[1], reverse=True)

# ---------------------------------------------------------
# 6. Print the sorted queries and their AP scores
# ---------------------------------------------------------
print("====================================================")
print("Queries sorted by Average Precision (AP):")

for q_id, score in final_ranking:
    print(f"Query {q_id}: {score:.4f}")