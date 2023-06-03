import os
import re
import string
from math import log10
from PyPDF2 import PdfReader

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


# reading vocabulary from file vocbulary.txt
vocabulary = {}
with open('vocabulary.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        ln = line.split(':')
        vocabulary[ln[1].strip()] = int(ln[0].strip())+1
    print('Total: ', len(lines), 'vocabulary items')


# Reading corrupt files list
with open('corruptFiles.txt', 'r') as f:
    corruptFiles = list(line.split(':')[1].strip() for line in f)
    print('Total: ', len(corruptFiles), 'corrupt files')


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
        docIds[name] = id
    print('Total: ', len(docIds), 'valid files')
    


# list of the folders containing pdf files
folders = ["C.A.", "C.M.A.", "C.P.", "Const.P.", "Crl.A.", "Crl.P.", "S.M.C."]

# iterating over the folders and pdf files and calculating raw term frequency
rawTermFreq = {}
for folder in folders:
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(".pdf") and filename not in corruptFiles:
            try:
                print("Processing:", filename + "...")
                # Open the PDF file and read its contents
                with open(os.path.join(folder, filename), 'rb') as f:
                    pdf = PdfReader(f)
                    text = ''
                    for page in pdf.pages:
                        text += page.extract_text()

                # Tokenize the text
                tokens = word_tokenize(text.lower())

                # Remove punctuation and stop words, alphanumeric, digits and words less than 3 characters
                table = str.maketrans('', '', string.punctuation)
                tokens = [word.translate(table) for word in tokens if len(word) > 2 and word not in stop_words and word.isalpha()]
                # list comprehension to filter out Urdu tokens
                tokens = [token for token in tokens if not urdu_pattern.search(token)]

                # Perform stemming and lemmatization
                stemmed_tokens = [stemmer.stem(word) for word in tokens]
                lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]

                # Update the rawTermFreq dictionary
                documentId = docIds[filename]
                for token in lemmatized_tokens:
                    tIndex = vocabulary[token]
                    if tIndex not in rawTermFreq:
                        rawTermFreq[tIndex] = {}
                        rawTermFreq[tIndex][documentId] = 1
                    else:
                        if documentId not in rawTermFreq[tIndex]:
                            rawTermFreq[tIndex][documentId] = 1
                        else:
                            rawTermFreq[tIndex][documentId] += 1
            except:
                print('Could not read file: ', filename)


print('----------------------------------')
# Save the Term Frequencies to rawTermFrequency.txt
print('\nWriting raw term frequencies of: ', len(rawTermFreq), 'tokens to rawTermFrequency.txt')
with open('rawTermFrequency.txt', 'w', encoding='utf-8', errors='ignore') as f:
    for tIndex, tFreq in sorted(rawTermFreq.items()):
        f.write(str(tIndex)+" ")
        for doc, freq in tFreq.items():
            f.write(f'{doc}:{freq} ')
        f.write('\n')
print('DONE WRITING RAW TERM FREQUENCY!\n...')


# Calculating log term frequency from raw term frequency and writing to logTermFrequency.txt
logTermFreq = {}
for tIndex, tFreq in rawTermFreq.items():
    log_freq_dict = {}
    for doc, freq in tFreq.items():
        log_freq_dict[doc] = 1 + log10(freq)
    logTermFreq[tIndex] = log_freq_dict


print('Writing log term frequencies of: ', len(logTermFreq), 'tokens to logTermFrequency.txt')
with open('logTermFrequency.txt', 'w', encoding='utf-8', errors='ignore') as f:
    for tIndex, tFreq in sorted(logTermFreq.items()):
        f.write(str(tIndex)+" ")
        for doc, freq in tFreq.items():
            f.write(f'{doc}:{freq} ')
        f.write('\n')
print('DONE WRITING LOG TERM FREQUENCY!\n...')


# Calculating IDF and writing to IDF.txt
idf = {}
N = len(docIds) # total number of documents
for tIndex, tFreq in rawTermFreq.items():
    df = len(tFreq)
    idf[tIndex] = log10(N/df)


print('Writing IDF of: ', len(idf), 'tokens to IDF.txt')
with open('IDF.txt', 'w', encoding='utf-8', errors='ignore') as f:
    for tIndex, tFreq in sorted(idf.items()):
        f.write(str(tIndex)+" "+str(tFreq))
        f.write('\n')
print('DONE WRITING IDF!\n...')


# Calculating TF-IDF and writing to TF-IDF.txt
tf_idf = {}
for tIndex, tFreq in logTermFreq.items():
    tfidf = {}
    for doc, freq in tFreq.items():
        tfidf[doc] = freq * idf[tIndex]
    tf_idf[tIndex] = tfidf


print('Writing TF-IDF of: ', len(tf_idf), 'tokens to TF-IDF.txt')
with open('TF-IDF.txt', 'w', encoding='utf-8', errors='ignore') as f:
    for tIndex, tFreq in sorted(tf_idf.items()):
        f.write(str(tIndex)+" ")
        for doc, freq in tFreq.items():
            f.write(f'{doc}:{freq} ')
        f.write('\n')
print('DONE WRITING TF-IDF!\n...')


# Calculating BM25 and writing to BM25.txt
bm_25 = {}
k = 0.5 # tuning parameter
for tIndex, tFreq in rawTermFreq.items():
    bm25 = {}
    df = len(tFreq)
    for doc, freq in tFreq.items():
        p1 = ((k+1) * int(freq)) / (int(freq) + k)
        bm25[doc] = p1 * log10((N+1)/df)
    bm_25[tIndex] = bm25


print('Writing BM25 of: ', len(bm_25), 'tokens to BM25.txt')
with open('BM25.txt', 'w', encoding='utf-8', errors='ignore') as f:
    for tIndex, tFreq in sorted(bm_25.items()):
        f.write(str(tIndex)+" ")
        for doc, freq in tFreq.items():
            f.write(f'{doc}:{freq} ')
        f.write('\n')
print('DONE WRITING BM25!\n...')
