from flask import Flask, render_template, request
import os, string, re, math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import base64
from io import BytesIO
# Following two lines should be uncommented if 'punkt' and 'stopwords' are not downloaded already
# nltk.download('punkt')
# nltk.download('stopwords')

# Number of search results to be displayed per page
MAX_RESULTS_PER_PAGE = 10


# an instance of SnowballStemmer for stemming
stemmer = SnowballStemmer('english')
# an instance of WordNetLemmatizer for lemmatization
lemmatizer = WordNetLemmatizer()
# a regex to match punctuation and urdu text
punct_pattern = re.compile(r'[^\w\s]')
urdu_pattern = re.compile('[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
# a set of stop words
stop_words = set(stopwords.words('english'))


# reading vocabulary from file vocbulary.txt
vocabulary = {}
with open('../vocabulary.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        ln = line.split(':')
        vocabulary[ln[1].strip()] = int(ln[0].strip())+1
    print('Total: ', len(lines), 'vocabulary items')


# reading normalized tfidf weights from file
normalized_tfidf = {}
with open("../normalized_tfidf.txt") as f:
    for line in f:
        parts = line.strip().split()
        index = int(parts[0])
        normalized_tfidf[index] = {}
        for subpart in parts[1:]:
            docId, normalized_tfidf_val = subpart.split(":")
            normalized_tfidf[index][docId] = float(normalized_tfidf_val)
    print('Total: ', len(normalized_tfidf), 'normalized tfidf')

# reading the raw term frequencies from file
rawTermFreq = {}
with open("../rawTermFrequency.txt", 'r') as f:
    for line in f:
        parts = line.strip().split()
        index = int(parts[0])
        rawTermFreq[index] = {}
        for subpart in parts[1:]:
            docId, rawtermfrequencies_val = subpart.split(":")
            rawTermFreq[index][docId] = int(rawtermfrequencies_val)
    print('Total: ', len(rawTermFreq), 'raw term frequencies')
    

# reading the idf from file
idf = {}
with open('../idf.txt', 'r') as f:
    for line in f:
        index, value = line.strip().split()
        idf[int(index)] = float(value)
    print('Total: ', len(idf), 'idf values')


# Function to remove punctuation marks
def remove_punctuation(text):
    punctuations = string.punctuation
    no_punct = "".join([char for char in text if char not in punctuations])
    return no_punct

# reading document index list for assigning IDs
docIds = {}
with open('../documentIndex.txt', 'r') as f:
    for line in f.readlines():
        if (line.startswith('Cat')) or len(line) < 1 or line == '\n':
            continue
        name = line.split(' ')[-1].strip()
        idString = name.split('____')[0].strip().split('-')
        id = remove_punctuation(idString[0])+idString[-1]
        docIds[id] = name
# print(docIds['SMC11'])

# getting abstract of documents from file with the help of another file ('links.txt')
docAbstracts_temp = []
with open('../description.txt', 'r') as f:
    for line in f:
        if (line.startswith('Cat')) or len(line) < 1 or line == '\n':
            continue
        line = re.sub(r'\s+', ' ', line)
        line = line.split(' ')
        desc = ' '.join(line[2:])
        docAbstracts_temp.append(desc)

docAbstracts = {}
with open('../links.txt', 'r') as f:
    i = 0
    for line in f:
        if line.startswith('Cat') or len(line) < 1 or line == '\n':
            continue
        pdfname = line.split('/')[-1]
        pdfname = pdfname.replace('\n','')
        for id, name in docIds.items():
            if pdfname in name:
                docAbstracts[id] = docAbstracts_temp[i]
                i += 1
# print(docAbstracts['SMC11'])

# getting rawTermFrequency for each docId
rtFreq = {}
for doc_id in docIds.keys():
    rtFreq[doc_id] = {}

for term_index, doc_freqs in rawTermFreq.items():
    for doc_id, freq in doc_freqs.items():
        if doc_id in docIds.keys():
            rtFreq[doc_id][term_index] = rawTermFreq[term_index][doc_id]
# print(rtFreq)


##########################
##  CODE FOR FLASK APP  ##
##########################

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    print(query)
    if len(query) < 1:
        return render_template('index.html')
    
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


    # Storing log terms frequencies in dictionary
    logTermFreq = {}
    for term, index in sorted(vocabulary.items()):
        term_dict = {}
        for query_id, freq_dict in sorted(tfQueries.items()):
            if index in freq_dict:
                term_freq = freq_dict[index]
                term_dict[query_id] = 1+math.log10(term_freq)
        if term_dict:
            logTermFreq[index] = term_dict
    logTermFreq = dict(sorted(logTermFreq.items()))
    # print(logTermFreq)

    # calculating TF-IDF for each term in each query
    tfidfQueries = {}
    for tIndex, tFreq in logTermFreq.items():
        _tfidf = {}
        for _query, freq in tFreq.items():
            _tfidf[_query] = freq * idf[tIndex]
        tfidfQueries[tIndex] = _tfidf
    # print(tfidfQueries)


    normalizedValues = {}
    for tIndex, tFreq in tfidfQueries.items():
        for idx, freq in tFreq.items():
            if idx not in normalizedValues:
                normalizedValues[idx] = 0
            normalizedValues[idx] += freq ** 2

    for queryId, normVal in normalizedValues.items():
        normalizedValues[queryId] = math.sqrt(normVal)
    # print('\n', normalizedValues)


    norm_tfidf_queries = {}
    for term_id, query_dict in tfidfQueries.items():
        norm_tfidf_queries[term_id] = {}
        for query_id, query_val in query_dict.items():
            norm_tfidf_queries[term_id][query_id] = query_val / normalizedValues[query_id]
    # print(norm_tfidf_queries)


    normalized_tfidfQueries = {}
    for term_id, query_dict in tfidfQueries.items():
        for query_id, tfidf_queries_val in query_dict.items():
            if query_id not in normalized_tfidfQueries:
                normalized_tfidfQueries[query_id] = {}
            normalized_tfidfQueries[query_id][term_id] = norm_tfidf_queries[term_id][query_id]
    # print('\n', normalized_tfidfQueries)

    queryIds = [1]

    # calculating similarity score of query against each document
    similarityScore = {}

    for qId in queryIds:
        similarityScore[qId] = {}
        for docId in docIds.keys():
            similarityScore[qId][docId] = 0

    for qId, q_tfidf in normalized_tfidfQueries.items():
        for tIdx, q_tfidf_val in q_tfidf.items():
            for doc_id, tfidf_val_doc in normalized_tfidf[tIdx].items():
                if doc_id in docIds.keys():
                    similarityScore[qId][doc_id] += q_tfidf_val * tfidf_val_doc

    topDocs = dict(sorted({k: v for k, v in similarityScore[1].items() if v != 0}.items(), key=lambda x: -x[1]))
    # print(topDocs)

    docDetails = []
    for docId, simScore in topDocs.items():
        folderName = re.sub(r'\d+','',docId)
        # building file path
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        file_path = os.path.join(parent_dir, folderName, docIds[docId])
        fileLink = f'<a href="{file_path}">{docIds[docId]}</a>'
        docDetails.append({'id': docId, 'link': fileLink, 'abstract': docAbstracts[docId]}) 
    
    # restricting search results to max 500
    if len(docDetails) > 500:
        docDetails = docDetails[:500]

    numOfPages = math.ceil(len(docDetails)/MAX_RESULTS_PER_PAGE)

    # check if page number is available
    page = request.args.get('page')
    if page:
        page = int(page)
    else:
        page = 1

    startIndex = (page-1) * MAX_RESULTS_PER_PAGE
    endIndex = startIndex + MAX_RESULTS_PER_PAGE

    resultsToShow = docDetails[startIndex:endIndex]

    # getting abstract from search results to generate wordcloud
    text = " ".join([d["abstract"] for d in docDetails])
    wc = WordCloud(width=600, height=400, background_color='white', relative_scaling=0.7, max_words=50).generate_from_text(text)
    img = BytesIO()
    wc.to_image().save(img, 'PNG')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()

    # rendering template and passing information to display in browser
    return render_template('search.html', query=query, results=resultsToShow, currentPage=page, numOfPages=numOfPages, img_data=img_base64)


if __name__ == '__main__':
    app.run(debug=True)
