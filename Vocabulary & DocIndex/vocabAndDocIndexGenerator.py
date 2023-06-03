import os
import re
import nltk
import string
from PyPDF2 import PdfReader
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')


# set to store unique tokens
vocabulary = {}

# an instance of SnowballStemmer for stemming
stemmer = SnowballStemmer('english')

# an instance of WordNetLemmatizer for lemmatization
lemmatizer = WordNetLemmatizer()

# a regex to match punctuation and urdu text
punct_pattern = re.compile(r'[^\w\s]')
urdu_pattern = re.compile('[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')

# a set of stop words
stop_words = set(stopwords.words('english'))

# list of the folders containing pdf files
folders = ["C.A.", "C.M.A.", "C.P.", "Const.P.", "Crl.A.", "Crl.P.", "S.M.C."]

# opening different files for writing data
corruptFiles = open("corruptFiles.txt", 'w')
documentIndex = open("documentIndex.txt", 'w')
documentIndex.write("Category".ljust(15) + "Index".ljust(8) + "Document Name\n")

# iterating over the folders and pdf files
index, cIdx = 0, 0
skipped = 0
for folder in folders:
    sNo = 1
    documentIndex.write("\n")
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(".pdf"):
            try:
                print("Indexing: ", filename)
                documentIndex.write(folder.ljust(15) + str(sNo).ljust(8) + filename + "\n")
                sNo += 1
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

                # Update the vocabulary dictionary/set
                for token in lemmatized_tokens:
                    if token not in vocabulary:
                        vocabulary[token] = index
                        index += 1
                    else:
                        skipped += 1
            except:
                # writing names of corrupt files to a text file
                cIdx += 1
                corruptFiles.write(str(cIdx) + ": " + filename + "\n")


print("\n\n" + str(skipped), " repeated tokens skipped!\n")
print("Writing ", len(vocabulary), " tokens to vocabulary.txt...\n")

# Save the vocabulary to vocabulary.txt
with open('vocabulary.txt', 'w', encoding='utf-8', errors='ignore') as f:
    for token, idx in vocabulary.items():
        f.write(f'{idx}: {token}\n')

print("Generated vocabulary written to \"vocabulary.txt!\"")
print("Files details written to \"documentIndex.txt!\"")
print("Corrupt files names written to \"corruptFiles.txt!\"\n")

