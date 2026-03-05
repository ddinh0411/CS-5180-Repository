#-------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 5180- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

# importing required libraries
import pandas as pd
import heapq
from sklearn.feature_extraction.text import CountVectorizer

# -----------------------------
# PARAMETERS
# -----------------------------
INPUT_PATH = "corpus/corpus.tsv"
BLOCK_SIZE = 100
NUM_BLOCKS = 10

READ_BUFFER_LINES_PER_FILE = 100
WRITE_BUFFER_LINES = 500


# ---------------------------------------------------------
# 1) READ FIRST BLOCK OF 100 DOCUMENTS USING PANDAS
# ---------------------------------------------------------
# Use pandas.read_csv with chunksize=100.
# Each chunk corresponds to one memory block.
# Convert docIDs like "D0001" to integers.
# ---------------------------------------------------------
# --> add your Python code here


# ---------------------------------------------------------
# 2) BUILD PARTIAL INDEX (SPIMI STYLE) FOR CURRENT BLOCK
# ---------------------------------------------------------
# - Use CountVectorizer(stop_words='english')
# - Fit and transform the 100 documents
# - Reconstruct binary postings lists from the sparse matrix
# - Store postings in a dictionary: term -> set(docIDs)
# ---------------------------------------------------------
# --> add your Python code here


# ---------------------------------------------------------
# 3) FLUSH PARTIAL INDEX TO DISK
# ---------------------------------------------------------
# - Sort terms lexicographically
# - Sort postings lists (ascending docID)
# - Write to: block_1.txt, block_2.txt, ..., block_10.txt
# - Format: term:docID1,docID2,docID3
# ---------------------------------------------------------
# --> add your Python code here


# ---------------------------------------------------------
# 4) REPEAT STEPS 1–3 FOR ALL 10 BLOCKS
# ---------------------------------------------------------
# - Continue reading next 100-doc chunks
# - After processing each block, flush to disk
# - Do NOT keep previous blocks in memory
# ---------------------------------------------------------
# --> add your Python code here


# ---------------------------------------------------------
# 5) FINAL MERGE PHASE
# ---------------------------------------------------------
# After all block files are created:
# - Open block_1.txt ... block_10.txt simultaneously
# ---------------------------------------------------------
# --> add your Python code here


# ---------------------------------------------------------
# 6) INITIALIZE READ BUFFERS
# ---------------------------------------------------------
# For each block file:
# - Read up to READ_BUFFER_LINES_PER_FILE lines
# - Parse each line into (term, postings_list)
# - Store in a per-file buffer
# ---------------------------------------------------------
# --> add your Python code here


# ---------------------------------------------------------
# 7) INITIALIZE MIN-HEAP (OR SORTED STRUCTURE)
# ---------------------------------------------------------
# - Push the first term from each buffer into a min-heap
# - Heap elements: (term, file_index)
# ---------------------------------------------------------
# --> add your Python code here


# ---------------------------------------------------------
# 8) MERGE LOOP
# ---------------------------------------------------------
# While heap is not empty:
#   1. Pop smallest term
#   2. Collect all buffers whose current term matches
#   3. Merge postings lists (sorted + deduplicated)
#   4. Advance corresponding buffer pointers
#   5. If a buffer is exhausted, read next 100 lines (if available)
# ---------------------------------------------------------
# --> add your Python code here


# ---------------------------------------------------------
# 9) WRITE BUFFER MANAGEMENT
# ---------------------------------------------------------
# - Append merged term-line to write buffer
# - If write buffer reaches WRITE_BUFFER_LINES:
#       flush (append) to final_index.txt
# - After merge loop ends:
#       flush remaining write buffer
# ---------------------------------------------------------
# --> add your Python code here


# ---------------------------------------------------------
# 10) CLEANUP
# ---------------------------------------------------------
# - Close all open block files
# - Ensure final_index.txt is properly written
# ---------------------------------------------------------
# --> add your Python code here