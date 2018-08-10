import pandas
import csv

from main_process.FeatureExtraction import Dokumen


def dataset_generate(filename, delim):
    tabel = pandas.read_csv(filename, header=0, delimiter=delim)
    dataset = dict()
    jumlah_data = len(tabel['Nama File'])
    for i in xrange(0, jumlah_data):
        nama_file = tabel['Nama File'][i]
        req_state = tabel['Requirement Statement'][i]
        noise = tabel['Noise?'][i]
        if nama_file not in dataset:
            dataset[nama_file] = []
        new_data = [req_state, noise]
        dataset[nama_file].append(new_data)

    return dataset


if __name__ == '__main__':
    # dataset = dataset_generate("../raw/data_translate.csv", ",")
    # dataset = dataset_generate("../raw/scoringmajority.csv", ";")
    dataset = dataset_generate("../raw/newdata.csv", ";")
    totalNoise = 0
    totalData = 0
    # with open("../raw/VectorFiturTranslate.csv", "wb") as csv_file:
    # with open("../raw/VectorFitur.csv", "wb") as csv_file:
    with open("../raw/tfidf.csv", "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',', dialect='excel')
        for nfile in dataset :
            sumNoise = 0
            sumData = 0
            print "Nama File : ", nfile, " - DONE"
            req = [row[0] for row in dataset[nfile]]
            label = [row[1] for row in dataset[nfile]]
            dokumen = Dokumen(nfile, req, label)
            sumNoise = sumNoise + dokumen.getNoise()
            sumData = sumData + len(label)
            totalData = totalData + sumData
            totalNoise = totalNoise + sumNoise
            print sumNoise
            print sumData
            dokumen.getFitur()
            # fiturs = dokumen.similarity
            # for fitur in fiturs :
            #     writer.writerow(fitur)
            for fitur in dokumen.token :
                writer.writerow(dokumen.token)
            fiturs = dokumen.matrix.todense()
            for j in range(0, dokumen.size) :
                tmp = fiturs[j].tolist()
                writer.writerow(tmp)
            # fiturs = dokumen.getFitur()
            # for fitur in fiturs :
            #     writer.writerow(fitur)
    print "----DONE----"
    print totalNoise
    print totalData
    # for nfile in dataset :
    #     print([row[1] for row in dataset[nfile]])
