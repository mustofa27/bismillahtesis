from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from main_process.PraprosesHelper import textPreprocess

import numpy

class Dokumen:
    def __init__(self, name, dokumen, label):
        self.name = name
        self.dokumen = dokumen
        self.fitur = [[0 for x in range(4)] for y in range(len(dokumen))]
        self.size = len(dokumen)
        self.label = label

    def textSetPreprocess(self) :
        for i in range(0, len(self.dokumen)) :
            self.dokumen[i] = textPreprocess(self.dokumen[i])

    def getFitur(self):
        self.textSetPreprocess()
        tfidf_vectorizer = TfidfVectorizer()
        # tfidf_vectorizer.smooth_idf = False
        self.matrix = tfidf_vectorizer.fit_transform(self.dokumen)
        self.similarity = numpy.zeros(shape=(self.size, self.size-1))
        for i in range(0, self.size):
            ind = 0
            for j in range(0, self.size):
                if i != j :
                    self.similarity[i][ind] = cosine_similarity(self.matrix[i],self.matrix[j])
                    ind += 1
        for i in range(0, self.size):
            self.fitur[i][0] = numpy.mean(self.similarity[i])
            self.fitur[i][1] = numpy.max(self.similarity[i])
            self.fitur[i][2] = numpy.std(self.similarity[i])
            self.fitur[i][3] = self.label[i]
        return self.fitur

    def getNoise(self):
        return numpy.sum(self.label)

if __name__ == '__main__':
    dok1 = ["i'm a student and lecturer freelance","you're a lecturer student","i'm just a student lecturer","you're a lecturer and student"]
    dokumen = Dokumen("blabla", dok1, [0, 0, 0, 0])
    print dokumen.getFitur()
    print dokumen.dokumen
