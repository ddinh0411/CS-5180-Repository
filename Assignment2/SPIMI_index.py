#-------------------------------------------------------------
# AUTHOR: Daniel Dinh
# FILENAME: SPIMI_index.py
# SPECIFICATION: takes the .tsv file from corpus makes 10 block files of inverted index and then merges all of them into generated final_index.txt
# FOR: CS 5180- Assignment #2
# TIME SPENT: 3-4 hours
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
# 4) REPEAT STEPS 1–3 FOR ALL 10 BLOCKS
# ---------------------------------------------------------
# - Continue reading next 100-doc chunks
# - After processing each block, flush to disk
# - Do NOT keep previous blocks in memory
# ---------------------------------------------------------

reader = pd.read_csv(INPUT_PATH, sep='\t', header=None, names=['docID', 'text'], chunksize=BLOCK_SIZE) # read csv with Pandas and set chunk size to 100

for i, block_chunk in enumerate(reader, start = 1): #iterate over 10 blocks
    # --- STEP 1 --- #
    block_chunk['docID'] =  block_chunk['docID'].str[1:].astype(int) # Convert all DocIDs to integers and remove D ('D0001 -> 0001)
    # --- STEP 2 --- #
    vectorizer = CountVectorizer(stop_words='english')
    block_matrix = vectorizer.fit_transform(block_chunk['text']) # train/transform on text from chunk of 100 docs
    terms = vectorizer.get_feature_names_out() # get terms / features

    block_terms = {} # get empty dictionary for terms
    rows, cols = block_matrix.nonzero() # gets non-sparse matrix for document-term

    for row,col in zip(rows, cols): # iterates through matrix
        term = terms[col] 
        doc_id = int(block_chunk['docID'].iloc[row]) # grab the document the term appears in
        if term not in block_terms: # if term not in dictionary yet
            block_terms[term] = set() # creates new set for term
        block_terms[term].add(doc_id) # adds document ID to term set
    # --- STEP 3 --- #
    with open(f"block_{i}.txt", "w") as f: # opens / creates new block file
        for term in sorted(block_terms): # sorts terms lexigraphically
            postings = sorted(block_terms[term]) # sorts the documents in the set in order
            postings_str = ",".join(map(str, postings)) # formats the string for block file
            f.write(f"{term}:{postings_str}\n") # writes to file

# ---------------------------------------------------------
# 5) FINAL MERGE PHASE
# ---------------------------------------------------------
# After all block files are created:
# - Open block_1.txt ... block_10.txt simultaneously
# ---------------------------------------------------------
# --> add your Python code here

block_files = [] # creates a set of block_files
for i in range(1, NUM_BLOCKS + 1): 
    f = open(f"block_{i}.txt", "r") # opens block file
    block_files.append(f) # adds opened file to block file list

# ---------------------------------------------------------
# 6) INITIALIZE READ BUFFERS
# ---------------------------------------------------------
# For each block file:
# - Read up to READ_BUFFER_LINES_PER_FILE lines
# - Parse each line into (term, postings_list)
# - Store in a per-file buffer
# ---------------------------------------------------------
# --> add your Python code here

read_buffers = {i: [] for i in range(NUM_BLOCKS)} # creating the read buffers of size 100

for i in range(NUM_BLOCKS): 
    for j in range(READ_BUFFER_LINES_PER_FILE):
        line = block_files[i].readline() # reads through and starts by reading the first 100 lines to the buffer from each block file
        if not line: # if file is under 100 lines break and exit loop (used later when refreshing buffer)
            break
        read_buffers[i].append(line.strip())

# ---------------------------------------------------------
# 7) INITIALIZE MIN-HEAP (OR SORTED STRUCTURE)
# ---------------------------------------------------------
# - Push the first term from each buffer into a min-heap
# - Heap elements: (term, file_index)
# ---------------------------------------------------------
# --> add your Python code here
active_posting = {} # collection of the active document list for each term in heap (used for merge loop)
heap = [] # min heap
heapq.heapify(heap)
for i in range(NUM_BLOCKS): # Pop first terms into heap
    line = read_buffers[i].pop(0) # pop lowest term in minheap
    term, post_str = line.split(":") # split line to grab term and posting list
    post_list = [int(x) for x in post_str.split(",")] # converts posting list to ints
    active_posting[i] = post_list # saves posting list for that term in active postings
    heapq.heappush(heap, (term, i)) # pushes just the term and the block it came from into heap

# ---------------------------------------------------------
# 8-9) MERGE LOOP /  WRITE BUFFER MANAGEMENT
# ---------------------------------------------------------
# - STEP 8 - #
# While heap is not empty:
#   1. Pop smallest term
#   2. Collect all buffers whose current term matches
#   3. Merge postings lists (sorted + deduplicated)
#   4. Advance corresponding buffer pointers
#   5. If a buffer is exhausted, read next 100 lines (if available)
# - STEP 9 - #
# - Append merged term-line to write buffer
# - If write buffer reaches WRITE_BUFFER_LINES:
#       flush (append) to final_index.txt
# - After merge loop ends:
#       flush remaining write buffer
# ---------------------------------------------------------
output_buffer = [] #sets up the output buffer 
current_term = None # sets variable to keep track of current term
current_posting = [] # sets list for the postings of the current term to be used for merge
open("final_index.txt", "w").close() # initializes new final_index.txt for writing 

while heap: # checks to see if there's still elements in the heap
    term, block_idx = heapq.heappop(heap) # pop a term and it's block id out

    # checks to see if read buffer is at end for popped term #
    if not read_buffers[block_idx]: # If block file is not empty
        for _ in range(READ_BUFFER_LINES_PER_FILE): # Overwrite read buffer with new 100 lines from block file
            line = block_files[block_idx].readline() # Grab line from file
            if not line: # if file is done then break
                break
            read_buffers[block_idx].append(line.strip())
    # grabs new term from buffer to replace the one just popped 
    if read_buffers[block_idx]: 
        next_line = read_buffers[block_idx].pop(0) # pop next term from read buffer
        next_term = next_line.split(":")[0]
        active_posting[block_idx] = [int(d) for d in next_line.split(":")[1].split(",")]
        heapq.heappush(heap, (next_term, block_idx)) # add new term to heap


    if term == current_term: # if there are still terms in the heap as the one being merged (ie multiple copies of 'apple')
        current_posting.extend(active_posting[block_idx]) # extend the current_postings with their postings lists as well
    else:
        if current_term is not None:
            final_posting = sorted(list(set(current_posting))) # format the postings lists into ascending order
            postings_str = ",".join(map(str, final_posting)) # format the string for printing to final_index.txt

            output_buffer.append(f"{current_term}:{postings_str}\n") # add the new term and its merged posting list to output buffer
            if len(output_buffer) >= WRITE_BUFFER_LINES: # if output buffer hits the limit for write buffer 500 lines
                with open("final_index.txt", "a") as f_out: # open output file and flushes the output buffer 
                    f_out.writelines(output_buffer)
                output_buffer = [] # resets the output buffer back to empty / 0
        current_term = term 
        current_posting = active_posting[block_idx].copy()

# handles last term in current_term
if current_term is not None: 
    final_posting = sorted(list(set(current_posting)))
    postings_str = ",".join(map(str, final_posting))
    output_buffer.append(f"{current_term}:{postings_str}\n")
# flushes whatever is left in output buffer out to final_index.txt
if output_buffer:
    with open("final_index.txt", "a") as f_out:
        f_out.writelines(output_buffer)
    output_buffer = []

# ---------------------------------------------------------
# 10) CLEANUP
# ---------------------------------------------------------
# - Close all open block files
# - Ensure final_index.txt is properly written
# ---------------------------------------------------------
# --> add your Python code here

# loops through and closes all block files
for f in block_files:
    f.close()