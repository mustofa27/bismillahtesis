from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph import laplacian
from kappa_coefficient import kappa_coef
import pandas
import numpy as np
import operator
import tmg_preprocessor as tmgp

def generate_matrixsim(matrix_tfidf) :
	matrix = matrix_tfidf.todense()
	n = len(matrix)
	matrix_sim = np.zeros((n,n))
	for i in range(n) :
		for j in range(i,n) :
			sim_score = cosine_similarity(matrix[i], matrix[j])
			matrix_sim[i][j] = sim_score
			matrix_sim[j][i] = sim_score
	return matrix_sim


if __name__ == "__main__" :
	text1 = "Ini bapak budi yang bernama andi"
	text2 = "Bapak andi bernama budi"
	text3 = "Sungai berasal dari sawah"
	text4 = "Sawah makmur karena sungai"
	text5 = "elon musk pergi ke bulan"
	text6 = "bulan tempat tinggal elon musk"
	text7 = "shitty but goldy"
	text8 = "goldy take a shitty jokes"
	text9 = "jancuk kowe cuk cuk"
	text10 = "cuk jancuk si pandai besi"

	textset = [text1, text2, text3, text4, text5, text6, text7, text8, text9, text10]
	tfidf_vectorizer = TfidfVectorizer()
	matrix_tfidf = tfidf_vectorizer.fit_transform(textset)
	
	#print(matrix_tfidf.todense())

	matrix_similarity = generate_matrixsim(matrix_tfidf)
	matrix_laplacian = laplacian(matrix_similarity, normed=True)
	#print(matrix_laplacian)
	eigval, eigvec = np.linalg.eig(matrix_laplacian)
	#print(eigval)
	
	idx = eigval.argsort()[::-1]   
	eigval = eigval[idx]
	eigvec = eigvec[:,idx]
	
	print(eigval)
	gaps = []
	'''
	for i in range(len(eigval)-1) :
		gap = eigval[i] - eigval[i+1]
		#gap = round(gap,2)
		print "gap : ", gap
		gaps.append(gap)
	'''

	largest_eigv = eigval[0]
	smallest_eigv = eigval[len(eigval)-1]
	k = largest_eigv - smallest_eigv
	k = int(k)

	#k = max(gaps)
	print "max : ",k
	