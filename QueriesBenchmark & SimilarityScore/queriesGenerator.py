import random

# reading tf-idf file
tfidf = {}
with open('TF-IDF.txt') as f:
    for line in f.readlines():
        parts = line.split()
        # Extract the key and values from the parts
        key = int(parts[0])
        values = {}
        for part in parts[1:]:
            subparts = part.split(":")
            values[subparts[0]] = float(subparts[1])
        tfidf[key]=values

# Initialize a dictionary to hold the sums of the inner dictionary values
tfidf_sums = {}
for key in tfidf.keys():
    # Get the inner dictionary for the current key
    inner_dict = tfidf[key]

    # Sum the values in the inner dictionary
    inner_sum = sum([float(value) for value in inner_dict.values()])

    # Store the sum as the value for the key in the sums dictionary
    tfidf_sums[key] = inner_sum

# Get the keys for the top 10 sums
tfidf_topKeys = sorted(tfidf_sums, key=tfidf_sums.get, reverse=True)[:10]


# reading bm25 file
bm25 = {}
with open('BM25.txt') as f:
    for line in f.readlines():
        parts = line.split()
        # Extract the key and values from the parts
        key = int(parts[0])
        values = {}
        for part in parts[1:]:
            subparts = part.split(":")
            values[subparts[0]] = float(subparts[1])
        bm25[key]=values

# Initialize a dictionary to hold the sums of the inner dictionary values
bm25_sums = {}
for key in bm25.keys():
    # Get the inner dictionary for the current key
    inner_dict = bm25[key]

    # Sum the values in the inner dictionary
    inner_sum = sum([float(value) for value in inner_dict.values()])

    # Store the sum as the value for the key in the sums dictionary
    bm25_sums[key] = inner_sum

# Get the keys for the top 10 sums
bm25_topKeys = sorted(bm25_sums, key=bm25_sums.get, reverse=True)[:10]


# reading vocabulary file
vocabulary = {}
with open('vocabulary.txt', encoding='utf-8') as f:
    for line in f.readlines():
        parts = line.strip().split(':')
        # Extract the key and values from the parts
        key = int(parts[0])
        value = str(parts[1].strip())
        vocabulary[key]=value

# getting terms with top 10 tfidf
tfidf_topTerms = {}
for key in tfidf_topKeys:
    tfidf_topTerms[key]= vocabulary[key]
# print(tfidf_topTerms.values())

# getting terms with top 10 bm25
bm25_topTerms = {}
for key in bm25_topKeys:
    bm25_topTerms[key]= vocabulary[key]
# print(bm25_topTerms.values())

# writing top terms to files
with open('top10_tfidf_terms.txt', 'w') as f:
    for val in tfidf_topTerms.values():
        f.write(val+'\n')

with open('top10_bm25_terms.txt', 'w') as f:
    for val in bm25_topTerms.values():
        f.write(val+'\n')

# creating a list of frequent terms for generating queries
frequent_terms = []
for val in tfidf_topTerms.values():
    if val not in frequent_terms:
        frequent_terms.append(val)

for val in bm25_topTerms.values():
    if val not in frequent_terms:
        frequent_terms.append(val)
print(frequent_terms)

# Generate five 2-word queries
twoWords_queries = []
for i in range(5):
    twoWords_queries.append(' '.join(random.sample(frequent_terms, 2)))
print(twoWords_queries)

# Generate five 3-word queries
threeWords_queries = []
for i in range(5):
    threeWords_queries.append(' '.join(random.sample(frequent_terms, 3)))
print(threeWords_queries)

# with open('twoWords_queries.txt', 'w') as f:
#     for val in twoWords_queries:
#         f.write(val+'\n')

# with open('threeWords_queries.txt', 'w') as f:
#     for val in threeWords_queries:
#         f.write(val+'\n')

# writing the generated 2-words and 3-words queries to file
queries = twoWords_queries + threeWords_queries
with open('queriesBenchmark.txt', 'w') as f:
    f.write('Query No.'.ljust(12)+'Query\n')
    for i in range(len(queries)):
        f.write(str(i+1).ljust(12)+queries[i]+'\n')