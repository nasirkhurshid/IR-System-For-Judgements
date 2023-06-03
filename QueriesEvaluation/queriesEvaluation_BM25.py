import os, string, re, math, pprint
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
# Following two lines should be uncommented if 'punkt' and 'stopwords' are not downloaded already
# nltk.download('punkt')
# nltk.download('stopwords')


# an instance of SnowballStemmer for stemming
stemmer = SnowballStemmer('english')
# an instance of WordNetLemmatizer for lemmatization
lemmatizer = WordNetLemmatizer()
# a regex to match punctuation and urdu text
punct_pattern = re.compile(r'[^\w\s]')
urdu_pattern = re.compile('[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
# a set of stop words
stop_words = set(stopwords.words('english'))


# reading queries file
queries = {}
with open('queries.txt') as f:
    for line in f.readlines():
        line = line.split()
        id = line[0]
        query = ' '.join(line[1:])
        queries[id] = query
# print(queries)

# true categories
trueCategories = ['CrlA','CrlP','CrlA','ConstP','CA','CMA','CP','CA','SMC','ConstP']

# reading vocabulary from file vocbulary.txt
vocabulary = {}
with open('vocabulary.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        ln = line.split(':')
        vocabulary[ln[1].strip()] = int(ln[0].strip())+1
    print('Total: ', len(lines), 'vocabulary items')


# reading normalized bm25 weights from file
normalized_bm25 = {}
with open("normalized_bm25.txt") as f:
    for line in f:
        parts = line.strip().split()
        index = int(parts[0])
        normalized_bm25[index] = {}
        for subpart in parts[1:]:
            docId, normalized_tfidf_val = subpart.split(":")
            normalized_bm25[index][docId] = float(normalized_tfidf_val)
    print('Total: ', len(normalized_bm25), 'normalized bm25')

# Function to remove punctuation marks
def remove_punctuation(text):
    punctuations = string.punctuation
    no_punct = "".join([char for char in text if char not in punctuations])
    return no_punct

# reading document index list for assigning IDs
docIds = {}
with open('documentIndex.txt', 'r') as f:
    for line in f.readlines():
        if (line.startswith('Cat')) or len(line) < 1 or line == '\n':
            continue
        name = line.split(' ')[-1].strip()
        idString = name.split('____')[0].strip().split('-')
        id = remove_punctuation(idString[0])+idString[-1]
        docIds[id] = name
# print(docIds['SMC11'])


#########################
# FUNCTIONS DEFINITIONS #
#########################

def recall(expected,result):
    recall = list()
    total = 0
    for r in result:
        if expected in r:
            total+=1
        recall.append(total/len(result))
    return recall[-1]


def precision(expected,result):
    precision = list()
    total = 0
    for i,r in enumerate(result):
        if expected in r:
            total+=1
        precision.append(round((total/(i+1)),3))
    return precision[-1]

def averagePercision(expected,result):
    average_percision = list()
    total = 0
    for i,r in enumerate(result):
        if expected in r:
            total+=1
            average_percision.append(round((total/(i+1)),3))
    return average_percision

def f1Score(recall,precision):
    f1_score = round(((2*recall*precision)/(recall+precision)),3)
    return f1_score



docsResults = []
docsCategories = []


for qid, query in queries.items():
    # Tokenize the text
    tokens = word_tokenize(query.lower())

    # Remove punctuation and stop words, alphanumeric, digits and words less than 3 characters
    table = str.maketrans('', '', string.punctuation)
    tokens = [word.translate(table) for word in tokens if len(word) > 2 and word not in stop_words and word.isalpha()]
    # list comprehension to filter out Urdu tokens
    tokens = [token for token in tokens if not urdu_pattern.search(token)]

    # Perform stemming and lemmatization
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
    print(lemmatized_tokens)
    tokens = lemmatized_tokens

    # Calculate term frequency (tf) for query
    tf_query = {}
    for term in tokens:
        term_index = vocabulary[term]
        if term_index in tf_query:
            tf_query[term_index] += 1
        else:
            tf_query[term_index] = 1

    tfQueries = {}
    tfQueries[1] = tf_query


    # Storing raw terms frequencies in dictionary
    rawTermFreq_queries = {}
    for term, index in sorted(vocabulary.items()):
        term_dict = {}
        for query_id, freq_dict in sorted(tfQueries.items()):
            if index in freq_dict:
                term_freq = freq_dict[index]
                term_dict[query_id] = term_freq
        if term_dict:
            rawTermFreq_queries[index] = term_dict
    rawTermFreq_queries = dict(sorted(rawTermFreq_queries.items()))
    # print(rawTermFreq_queries)

    normalizedValues = {}
    for tIndex, tFreq in rawTermFreq_queries.items():
        for idx, freq in tFreq.items():
            if idx not in normalizedValues:
                normalizedValues[idx] = 0
            normalizedValues[idx] += freq ** 2

    for queryId, normVal in normalizedValues.items():
        normalizedValues[queryId] = math.sqrt(normVal)
    # print('\n', normalizedValues)

    norm_queries = {}
    for term_id, query_dict in rawTermFreq_queries.items():
        norm_queries[term_id] = {}
        for query_id, query_val in query_dict.items():
            norm_queries[term_id][query_id] = query_val / normalizedValues[query_id]
    # print(norm_queries)


    normalizedQueries = {}
    for term_id, query_dict in rawTermFreq_queries.items():
        for query_id, queries_val in query_dict.items():
            if query_id not in normalizedQueries:
                normalizedQueries[query_id] = {}
            normalizedQueries[query_id][term_id] = norm_queries[term_id][query_id]
    # print('\n', normalizedQueries)

    queryIds = [1]

    # calculating similarity score of query against each document
    similarityScore = {}

    for qId in queryIds:
        similarityScore[qId] = {}
        for docId in docIds.keys():
            similarityScore[qId][docId] = 0

    for qId, q_norm in normalizedQueries.items():
        for tIdx, q_norm_val in q_norm.items():
            for doc_id, bm25_val_doc in normalized_bm25[tIdx].items():
                if doc_id in docIds.keys():
                    similarityScore[qId][doc_id] += q_norm_val * bm25_val_doc

    topDocs = dict(sorted({k: v for k, v in similarityScore[1].items() if v != 0}.items(), key=lambda x: -x[1]))
    # print(topDocs)

    docDetails = []
    for docId, simScore in topDocs.items():
        category = re.sub(r'\d+','',docId)
        filename = docIds[docId]
        docDetails.append({'category': category, 'filename': filename}) 

    docsResults.append(list(docDetails)[:10])


for results in docsResults:
    categories = []
    for diction in results:
        categories.append(diction['category'])
    docsCategories.append(categories)

index = 0
mean_average_percision = 0
lines = []
for topCats in docsCategories:
    recallVal = recall(trueCategories[index].upper(),topCats)
    precisionVal = precision(trueCategories[index].upper(),topCats)
    f1_score = f1Score(recallVal,precisionVal)
    average_percision = averagePercision(trueCategories[index].upper(),topCats)
    final_average_percision = sum(average_percision)/len(average_percision)
    mean_average_percision += final_average_percision
    line = ('Query Term-'+ str(index+1)).ljust(18)+'BM25'.ljust(12)+str(precisionVal).ljust(6)+str(recallVal).ljust(6)+str(f1_score).ljust(6)+str(round(final_average_percision,3)).ljust(6)+'\n'
    lines.append(line)
    index+=1
mean_average_percision = mean_average_percision/10

with open('bm25Evaluation.txt', 'w') as f:
    f.write('Query Terms'.ljust(18)+'Weighting'.ljust(12)+'P'.ljust(6)+'R'.ljust(6)+'F'.ljust(6)+'AP'.ljust(6)+'\n')
    for line in lines:
        f.write(line)
    f.write(f'Mean Average Precision for 10 queries = {mean_average_percision}\n')

print('\nEvaluation Data written to bm25Evaluation.txt\n')