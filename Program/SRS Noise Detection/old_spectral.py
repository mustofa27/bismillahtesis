from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralClustering
import pandas
import operator
import tmg_preprocessor as tmgp

if __name__ == '__main__' :
	
	# Ambil data dari file .csv
	req_data = pandas.read_csv("req_set.csv", header=0, delimiter="\t", quoting=3)
	row, col = req_data.shape
	print "Jumlah data : ", row, "\n"

	#Ambil index noise
	noise_indexes = []
	for i in xrange(0,row) :
		if req_data['Noise?'][i] == 1 :
			noise_indexes.append(i)

	#Pra-pemrosesan kata
	preprocessed_dataset = tmgp.textSetPreprocess(req_data['Requirement Statement'])

	#Membuat matrix tf-idf dari data requirement statement
	tfidf_vectorizer = TfidfVectorizer(max_df=0.5)
	matrix_tfidf = tfidf_vectorizer.fit_transform(preprocessed_dataset)

	#Implementasi spectral clustering 
	#n = jumlah cluster
	n = 6
	spectral = SpectralClustering(n_clusters=n, \
									affinity="nearest_neighbors", \
									n_neighbors=1 )
	cluster_result = spectral.fit_predict(matrix_tfidf)
	print "Hasil Spectral Clustering : \n", cluster_result, "\n"
	
	#Jumlah anggota tiap cluster disimpan 
	cluster_sum = dict()
	for i in xrange(0,n) :
		cluster_sum[i] = len([c for c in cluster_result if c==i])
		print "Cluster-",i, " : ", cluster_sum[i]
	
	#Sorting cluster berdasarkan jumlah anggota cluster
	cluster_sum = sorted(cluster_sum.items(), key=operator.itemgetter(1))
	min_number = cluster_sum[0][1]

	#Mencari cluster dengan jumlah paling sedikit
	smallest_cluster = []
	for data in cluster_sum :
		if data[1] == min_number :
			smallest_cluster.append(data[0])
	
	print "Smallest cluster = ", smallest_cluster, "\n"

	print "Predicted Noise : "
	idx = 0
	for cluster in cluster_result :
		if cluster in smallest_cluster :
			print req_data['Requirement Statement'][idx]
		idx = idx + 1
	print ""

	#Cek apakah cluster noise cocok dengan ground truth
	print "Matched Real Noises : "
	for i in xrange(0,len(noise_indexes)) :
		print req_data['Requirement Statement'][noise_indexes[i]]
		if cluster_result[noise_indexes[i]] in smallest_cluster :
			print "MATCH"
		else :
			print "NOT MATCH"
	