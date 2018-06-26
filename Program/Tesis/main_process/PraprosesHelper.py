'''
- modul ini intinya untuk membersihkan teks (preprocessing) dokumen
- terdapat juga fungsi untuk membersihkan kumpulan dokumen teks
'''
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

def textPreprocess(text) :
	#1. remove non-letters
	text = re.sub("[^a-zA-Z]", " ", text)

	#2. convert to lowercase & tokenization
	#tokenizer from CountVectorizer
	#omit word that only 1 letter
	tokenizer = CountVectorizer().build_tokenizer()
	words = tokenizer(text.lower())

	#3. remove the stopwords & stemming
	stemmer = PorterStemmer()
	stoplist = set(stopwords.words("english"))
	words_preprocessed = [stemmer.stem(w) for w in words if not w in stoplist]

	#4. join the cleaned words
	return (" ".join(words_preprocessed))
