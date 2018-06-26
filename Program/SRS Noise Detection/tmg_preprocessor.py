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

'''
'''
def textSetPreprocess(textSet) :
	textSet_preprocessed = []
	for text in textSet :
		text = textPreprocess(text)
		textSet_preprocessed.append(text)
	return textSet_preprocessed

if __name__ == '__main__' :
	text = "The statistics produced and displayed in the Term-Document Matrix contain basic information on the frequency of terms appearing in the document collection"
	text_preprocessed = textPreprocess(text)
	print text, "\n"
	print text_preprocessed

	textSet = [
	"submit jobs with the associated deadline, cost, and execution time",
	"query the cluster to establish the current cost per unit time for submitting new jobs",
	"monitor the status of submitted jobs",
	"cancel jobs submitted by him",
	"check his credit balance",
	]
	textSet_preprocessed = textSetPreprocess(textSet)
	print textSet, "\n"
	print textSet_preprocessed
