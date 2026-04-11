#-------------------------------------------------------------------------
# AUTHOR: Daniel Dinh
# FILENAME: vsm_search.py
# SPECIFICATION: description of the program
# FOR: CS 5180- Assignment #3
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

# importing required libraries
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# 1. Load the input files
# ---------------------------------------------------------

# Saves all data files as Dataframes
docs_collection = pd.read_csv('docs.csv', header=0)
queries = pd.read_csv('queries.csv', header=0)
judgement_list = pd.read_csv('relevance_judgments.csv', header=0)

# ---------------------------------------------------------
# 2. Build the TF-IDF matrix for the documents
# ---------------------------------------------------------
# Requirement: remove stopwords only

tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
tfidf_train = tfidf_vectorizer.fit_transform(docs_collection['text'])

# ---------------------------------------------------------
# 3. Process each query and compute AP values
# ---------------------------------------------------------
# Making of the tf-idf model and also getting cosine scores 
tfidf_test = tfidf_vectorizer.transform(queries['query_text'])

cosine_results = []
document_ids = docs_collection['doc_id'].values

for i in range(len(queries)):
    query_vector = tfidf_test[i]

    cosine_score = cosine_similarity(query_vector, tfidf_train).flatten()

    cosine_rankings = np.argsort(cosine_score)[::-1]
    doc_rankings = document_ids[cosine_rankings]
    
    cosine_results.append(doc_rankings)

# -----------------------------------------------------
# 4. Compute Average Precision (AP)
# -----------------------------------------------------

# Logic
# Uses Pandas and iloc/loc to grab specific rows that are relevant
# Outer loop is done by grabbing specific rows from judgement table for query_id
    # Grab query_id with formatting from query listr
    # Grab rows from judgement table with same query_id
    # Grab all rows that are deemed "relevant" and count
# Inner loops through each query_doc list
    # Checks if at given doc_id for query_id is 'R' then add 1 to hit
    # adds precision for rank and hit and adds to total for query
# Total up precision by total relevance for query
# Add to ap scores list

ap_scores = []
query_ids = queries['query_id'].values

for q_id, ranked_docs in enumerate(cosine_results):
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
# Ranking the queries by descending ap scores
final_ranking = sorted(zip(query_ids, ap_scores), key=lambda x: x[1], reverse=True)

# ---------------------------------------------------------
# 6. Print the sorted queries and their AP scores
# ---------------------------------------------------------
# Print out the queries and ap scores
print("====================================================")
print("Queries sorted by Average Precision (AP):")
for q_id, score in final_ranking:
    print(f"Query {q_id}: {score:.4f}")
