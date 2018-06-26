from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralClustering
from kappa_coefficient import kappa_coef
import pandas
import csv
import operator
import tmg_preprocessor as tmgp

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
			#Memasukan hasil prediksi positive noise ke variabel
			predicted_noise.append(0)
	
	#print "Perbandingan noise : "
	#print noise_req
	#print predicted_noise
	kc = kappa_coef(noise_req, predicted_noise)
	#score_data = [n, kc]
	#print score_data

	return kc


if __name__ == '__main__' :

	#organisir data
	dataset1 = dataset_generate("scoringgugik.csv")
	dataset2 = dataset_generate("scoringpatricia.csv")

	global_kappa_scores = []
	kappa_score_sum = 0

	#proses data per file
	for nfile in dataset1 :
		
		#menyimpan semua requirement pada suatu file
		all_req1 = []
		all_req2 = []

		#menyimpan status noise semua requirement pada suatu file
		noise_req1 = []
		noise_req2 = []

		#pembagian dataset
		for rstate in dataset1[nfile] :
			#print rstate[0], " ", rstate[1]
			all_req1.append(rstate[0])
			noise_req1.append(rstate[1])

		for rstate in dataset2[nfile] :
			#print rstate[0], " ", rstate[1]
			all_req2.append(rstate[0])
			noise_req2.append(rstate[1])

		kappa_score = kappa_coef(noise_req1, noise_req2)
		kappa_score_sum = kappa_score_sum + kappa_score
		global_kappa_scores.append([nfile, kappa_score])
	
	kappa_score_average = kappa_score_sum / float(len(dataset1))

	#masukin hasil ke file csv
	with open("Hasil.csv", "wb") as csv_file:
	        writer = csv.writer(csv_file, delimiter=';', dialect='excel')
	        for line in global_kappa_scores:
	            writer.writerow(line)
	      	writer.writerow(['average', kappa_score_average ])

	print "----DONE----"




