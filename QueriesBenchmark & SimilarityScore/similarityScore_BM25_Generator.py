from math import log10, sqrt
from nltk.tokenize import word_tokenize
import string

# reading bm25 file
BM25 = {}
with open('BM25.txt') as f:
    for line in f.readlines():
        parts = line.split()
        # Extract the key and values from the parts
        key = int(parts[0])
        values = {}
        for part in parts[1:]:
            subparts = part.split(":")
            values[subparts[0]] = float(subparts[1])
        BM25[key]=values

        
# reading queries file
queries = {}
with open('queriesbenchmark.txt') as f:
    next(f)
    for line in f.readlines():
        line = line.split()
        id = line[0]
        query = ' '.join(line[1:])
        queries[id] = query
# print(queries)

# reading vocabulary from file vocbulary.txt
vocabulary = {}
with open('vocabulary.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        ln = line.split(':')
        vocabulary[ln[1].strip()] = int(ln[0].strip())+1
# print(vocabulary)

# tokenizing the queries
tokenized_queries = {}
for query_id, query_text in queries.items():
    tokenized_queries[query_id] = word_tokenize(query_text)
# print(tokenized_queries)


# Calculating term frequency for each query
tfQueries = {}
for query_id, tokens in tokenized_queries.items():
    tf_query = {}
    for term in tokens:
        term_index = vocabulary[term]
        if term_index in tf_query:
            tf_query[term_index] += 1
        else:
            tf_query[term_index] = 1
    tfQueries[query_id] = tf_query
# print(tfQueries)


# Storing raw terms frequencies in dictionary
rawTermFreq = {}
for term, index in sorted(vocabulary.items()):
    term_dict = {}
    for query_id, freq_dict in sorted(tfQueries.items()):
        if index in freq_dict:
            term_freq = freq_dict[index]
            term_dict[query_id] = term_freq
    if term_dict:
        rawTermFreq[index] = term_dict

rawTermFreq = dict(sorted(rawTermFreq.items()))
# print(rawTermFreq)


normalizedValues = {}
for tIndex, tFreq in rawTermFreq.items():
    for idx, freq in tFreq.items():
        if idx not in normalizedValues:
            normalizedValues[idx] = 0
        normalizedValues[idx] += freq ** 2

for queryId, normVal in normalizedValues.items():
    normalizedValues[queryId] = sqrt(normVal)
# print('\n', normalizedValues)


norm_queries = {}
for term_id, query_dict in rawTermFreq.items():
    norm_queries[term_id] = {}
    for query_id, query_val in query_dict.items():
        norm_queries[term_id][query_id] = query_val / normalizedValues[query_id]
# print(norm_queries)


normalizedQueries = {}
for term_id, query_dict in rawTermFreq.items():
    for query_id, queries_val in query_dict.items():
        if query_id not in normalizedQueries:
            normalizedQueries[query_id] = {}
        normalizedQueries[query_id][term_id] = norm_queries[term_id][query_id]
# print('\n', normalizedQueries)

# getting unique query ids
queryIds = []
for term_index in norm_queries:
    for query_id in norm_queries[term_index]:
        if query_id not in queryIds:
            queryIds.append(query_id)
queryIds = sorted(queryIds)
# print(queryIds)


# normalizing original bm25 for checking similarity
normalized_dict = {}
for term_id, doc_dict in BM25.items():
    for doc_id, bm25_val in doc_dict.items():
        if doc_id not in normalized_dict:
            normalized_dict[doc_id] = 0
        normalized_dict[doc_id] += bm25_val ** 2

for doc_id, norm_val in normalized_dict.items():
    normalized_dict[doc_id] = sqrt(norm_val)

normalized_bm25 = {}
for term_id, doc_dict in BM25.items():
    normalized_bm25[term_id] = {}
    for doc_id, bm25_val in doc_dict.items():
        normalized_bm25[term_id][doc_id] = bm25_val / normalized_dict[doc_id]

# writing normalized bm25 to file
with open('normalized_bm25.txt', 'w') as f:
    for index, term_dict in normalized_bm25.items():
        line = f"{index}"
        for doc_id, freq in sorted(term_dict.items()):
            line += f" {doc_id}:{freq}"
        f.write(line + "\n")


# Function to remove punctuation marks
def remove_punctuation(text):
    punctuations = string.punctuation
    no_punct = "".join([char for char in text if char not in punctuations])
    return no_punct

# Reading document index list for assigning IDs
docIds = {}
with open('documentIndex.txt', 'r') as f:
    for line in f.readlines():
        if (line.startswith('Cat')) or len(line) < 1 or line == '\n':
            continue
        name = line.split(' ')[-1].strip()
        idString = name.split('____')[0].strip().split('-')
        id = remove_punctuation(idString[0])+idString[-1]
        docIds[id] = name
# print(docIds)

# calculating similarity score of each query against each document
similarityScore = {}

for qId in queryIds:
    similarityScore[qId] = {}
    for docId in docIds.keys():
        similarityScore[qId][docId] = 0

for qId, q_tf in normalizedQueries.items():
    for tIdx, q_tf_val in q_tf.items():
        for doc_id, bm25_val_doc in normalized_bm25[tIdx].items():
            if doc_id in docIds.keys():
                similarityScore[qId][doc_id] += q_tf_val * bm25_val_doc

# writing similarity score to file
with open("similarityScore_BM25.txt", "w") as f:
    for qId, docSim in similarityScore.items():
        f.write(f"Query Term {qId}\n")
        f.write("Weighting Scheme: BM25\n")
        for doc_id, similarity in docSim.items():
            f.write(f"{str(doc_id).ljust(10)} {str(docIds[doc_id]).ljust(60)} {similarity}\n")
        f.write("\n")

# extracting top 10 similarities from all the similarities
top_10_similarities = {}

for qId in similarityScore:
    docSim = similarityScore[qId]
    top_similarities = sorted(docSim.items(), key=lambda x: x[1], reverse=True)[:10]
    topDocs = {doc_id: similarity for doc_id, similarity in top_similarities}
    top_10_similarities[qId] = topDocs

with open("top10_similarityScore_BM25.txt", "w") as f:
    for qId in sorted(top_10_similarities.keys(), key=lambda x: int(x)):
        f.write(f"Query Term {qId}\n")
        f.write("Weighting Scheme: BM25\n")
        for doc_id, similarity in top_10_similarities[qId].items():
            f.write(f"{str(doc_id).ljust(10)} {str(docIds[doc_id]).ljust(60)} {similarity}\n")
        f.write("\n")
