
def kappa_coef(rater1, rater2) :
	n = len(rater1)

	yy = yn = ny = nn = 0
	#mencari data masukan : yy, yn, ny, nn
	for i in xrange(0, n) :
		if rater1[i] == 1 and rater2[i] == 1 :
			yy = yy + 1
		elif rater1[i] == 1 and rater2[i] == 0 :
			yn = yn + 1
		elif rater1[i] == 0 and rater2[i] == 1 :
			ny = ny + 1
		else :
			nn = nn + 1
	
	total_rating = float(yy + yn + ny + nn)
	po = (yy + nn) / total_rating
	py = ((yy + yn) / total_rating) * ((yy + ny) / total_rating) 
	pn = ((nn + yn) / total_rating) * ((nn + ny) / total_rating)
	pe = py + pn
	k = (po - pe) / (1 - pe)

	return round(k,4)


if __name__ == '__main__' :

	rater1 = [1, 0, 0, 1]
	rater2 = [1, 1, 0, 0]
	k = kappa_coef(rater1, rater2)
	print "Kappa : ", k