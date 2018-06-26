'''
v2 - Jumlah klasternya ditentukan secara greedy.
'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralClustering
from kappa_coefficient import kappa_coef
import pandas
import csv
import operator
import tmg_preprocessor as tmgp

'''
Subroutin ini memisahkan kolom - kolom pada dataset .csv
menjadi sebuah variabel berbentuk dictionary.
'''
def dataset_generate(filename) :

	tabel = pandas.read_csv(filename, header=0, delimiter=";")
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

	return dataset

'''
Subroutin ini mencari noise pada requirement menggunakan
klasterisasi spektral, dan mencocokannya dengan penilaian
noise yang telah dilakukan secara manual. 

Hasil akhir dari subroutin ini adalah nilai koefisiensi 
kappa antara hasil prediksi noise dan penilaian noise secara
manual.
'''
def noise_detect(jml_req, n, matrix_tfidf, noise_req) :

	spectral = SpectralClustering(n_clusters=n)
	cluster_result = spectral.fit_predict(matrix_tfidf)
	
	#Jumlah anggota tiap cluster disimpan 
	cluster_sum = dict()
	for i in xrange(0,n) :
		cluster_sum[i] = len([c for c in cluster_result if c==i])
	
	#Sorting cluster berdasarkan jumlah anggota cluster
	cluster_sum = sorted(cluster_sum.items(), key=operator.itemgetter(1))
	min_number = cluster_sum[0][1]

	#Mencari cluster dengan jumlah paling sedikit
	smallest_cluster = []
	for data in cluster_sum :
		if data[1] == min_number :
			smallest_cluster.append(data[0])
	
	#masukan hasil prediksi noise
	predicted_noise = []
	for i in xrange(0, jml_req) :
		if (cluster_result[i] in smallest_cluster) :
			predicted_noise.append(1)
		else :
			predicted_noise.append(0)
	
	#perhitungan nilai koefisien kappa
	kc = kappa_coef(noise_req, predicted_noise)
	return kc


if __name__ == '__main__' :

	dataset = dataset_generate("scoringmajority.csv")

	global_kappa_scores = []
	kappa_score_sum = 0
	jumlah_dataset = len(dataset)

	#proses data per file
	for nfile in dataset :
		
		print "Nama File : ", nfile, " - DONE"

		#menyimpan semua requirement pada suatu file
		all_req = []
		jumlah_req = len(dataset[nfile])

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
		#memeriksa semua kemungkinan jumlah klaster
		jml_req = len(all_req)
		kappa_scores = dict()
		for n in xrange(2, jml_req) :
			noise_score = noise_detect(jml_req, n, matrix_tfidf, noise_req)
			kappa_scores[n] = noise_score

		#cari nilai kappa tertinggi
		sorted_kappa_scores = sorted(kappa_scores.items(), key=operator.itemgetter(1))
		n = len(sorted_kappa_scores)
		max_kappa_n = sorted_kappa_scores[n-1][0]
		max_kappa_score = sorted_kappa_scores[n-1][1]
		kappa_score_sum = kappa_score_sum + max_kappa_score 
		global_kappa_scores.append([nfile, jumlah_req, max_kappa_n, max_kappa_score])

	#menghitung nilai rata - rata kappa
	kappa_score_average = kappa_score_sum / float(jumlah_dataset)
	
	#masukin hasil ke file csv
	with open("Hasil.csv", "wb") as csv_file:
	        writer = csv.writer(csv_file, delimiter=';', dialect='excel')
	        for line in global_kappa_scores:
	            writer.writerow(line)
	      	writer.writerow(['','','average', kappa_score_average ])

	print "----DONE----"




