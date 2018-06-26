from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralClustering
from kappa_coefficient import kappa_coef
import pandas
import operator
import tmg_preprocessor as tmgp

if __name__ == '__main__' :

	tabel = pandas.read_csv("req_set.csv", header=0, delimiter="\t", quoting=3)
	
	'''
	print "dataset size : ", tabel.shape
	print "dataset (nama file) size : ", tabel['Nama File'].shape
	print len(tabel['Nama File'])
	'''

	#organisir data
	dataset = dict()
	jumlah_data = len(tabel['Nama File'])
	for i in xrange(0, jumlah_data) :
		nama_file = tabel['Nama File'][i]
		req_state = tabel['Requirement Statement'][i]
		noise = tabel['Noise?'][i]
		if nama_file not in dataset :
			dataset[nama_file] = []
		new_data = [req_state, noise]
		dataset[nama_file].append(new_data)

	#proses data per file
	for nfile in dataset :
		
		print "Nama File : ", nfile, "\n"

		#menyimpan semua requirement pada suatu file
		all_req = []

		#menyimpan status noise semua requirement pada suatu file
		noise_req = []

		#pembagian dataset
		for rstate in dataset[nfile] :
			#print rstate[0], " ", rstate[1]
			all_req.append(rstate[0])
			noise_req.append(rstate[1])

		#Pra-pemrosesan kata
		preprocessed_dataset = tmgp.textSetPreprocess(all_req)

		#Membuat matrix tf-idf dari data requirement statement
		tfidf_vectorizer = TfidfVectorizer()
		matrix_tfidf = tfidf_vectorizer.fit_transform(preprocessed_dataset)

		#Implementasi spectral clustering 
		#n = jumlah cluster
		n = 11
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

		print "Predicted Noise vs Ground Truth: "
		predicted_noise = []
		jml_req = len(cluster_result)
		for i in xrange(0, jml_req) :
			if (cluster_result[i] in smallest_cluster) :
				print all_req[i]
				if noise_req[i] == 1 :
					print 'MATCH'
				else :
					print 'NOT MATCH'
				#Memasukan hasil prediksi positive noise ke variabel
				predicted_noise.append(1)
			else :
				#Memasukan hasil prediksi positive noise ke variabel
				predicted_noise.append(0)
		print " "
		
		print "Perbandingan noise : "
		print noise_req
		print predicted_noise
		print "Kappa : ", kappa_coef(noise_req, predicted_noise), "\n"


