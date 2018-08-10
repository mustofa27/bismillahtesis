from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from main_process.PraprosesHelper import textPreprocess

import numpy

class Dokumen:
    def __init__(self, name, dokumen, label):
        self.name = name
        self.dokumen = dokumen
        self.fitur = [[0 for x in range(8)] for y in range(len(dokumen))]
        self.size = len(dokumen)
        self.label = label
        self.textSetPreprocess()
        tfidf_vectorizer = TfidfVectorizer()
        # tfidf_vectorizer.smooth_idf = False
        self.matrix = tfidf_vectorizer.fit_transform(self.dokumen)
        self.token = tfidf_vectorizer.get_feature_names()

    def textSetPreprocess(self) :
        for i in range(0, len(self.dokumen)) :
            self.dokumen[i] = textPreprocess(self.dokumen[i])

    def getFitur(self):
        self.similarity = numpy.zeros(shape=(self.size, self.size-1))
        for i in range(0, self.size):
            ind = 0
            for j in range(0, self.size):
                if i != j :
                    self.similarity[i][ind] = cosine_similarity(self.matrix[i], self.matrix[j])
                    ind += 1
        for i in range(0, self.size):
            self.fitur[i][0] = numpy.mean(self.similarity[i])
            self.fitur[i][1] = numpy.max(self.similarity[i])
            self.fitur[i][2] = numpy.std(self.similarity[i])
            self.fitur[i][3] = numpy.min(self.similarity[i])
            self.fitur[i][4] = numpy.var(self.similarity[i])
            self.fitur[i][5] = self.label[i]
            self.fitur[i][6] = self.name
            self.fitur[i][7] = self.dokumen[i]
        return self.fitur

    def getNoise(self):
        return numpy.sum(self.label)

    def jaccard_similarity(self, arr, arr1):
        min = 0
        max = 0
        for i in range(0, arr.shape[1]):
            if arr[0,i] < arr1[0,i]:
                min = min + arr[0,i]
                max = max + arr1[0,i]
            else :
                min = min + arr1[0,i]
                max = max + arr[0,i]
        sim = min / max
        return sim

if __name__ == '__main__':
    dok1 = ["i'm a student and lecturer freelance","you're a lecturer student","i'm just a student lecturer","you're a lecturer and student"]
    dokumen = Dokumen("blabla", dok1, [0, 0, 0, 0])
    dokumen1 = Dokumen("blabla", dok1, [0, 0, 0, 0])
    # if dokumen.matrix[0][0] < dokumen1.matrix[0][0] :
    print dokumen.matrix[0,1]
    print dokumen.dokumen
