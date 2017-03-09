import numpy
import scipy
import sys
import time
import getopt
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_selection import VarianceThreshold 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFpr

# Reading a bag of words file back into python. The number and order
# of sentences should be the same as in the *samples_class* file.
def read_bagofwords_dat(myfile):
  bagofwords = numpy.genfromtxt(myfile, delimiter=',')
  return bagofwords

def read_class_values_dat(myfile):
	classes = numpy.genfromtxt(myfile, delimiter='\n')
	return classes

def write_probs_to_file(myfile, mydata):
	outfile= open(myfile, 'w')
	numpy.savetxt(outfile, mydata, '%.4f', ', ', '\n')
	outfile.close()

def write_classes_to_file(myfile, mydata):
	outfile= open(myfile, 'w')
	numpy.savetxt(outfile, mydata, '%d', ', ', '\n')
	outfile.close()

def main(argv):
	
	start_time = time.time()
	print "running main()"

	bowfile = ''
	clsfile = ''
	tstfile = ''

	# Parse arguments
	try:
		opts, args = getopt.getopt(argv,"b:c:t:T:k:a:",["bow=","cls=","tst=","tstcls=","k=","alpha="])
	except getopt.GetoptError:
		print 'Usage: \n python naiveBayes.py -b <bagofwords_csv> -c <classes_txt> -t <tst_bagofwords_csv> -T <tst_classes_txt> -k <kth_best> -a <alpha>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'Usage: \n python naiveBayes.py -b <bagofwords_csv> -c <classes_txt> -t <tst_bagofwords_csv> -T <tst_classes_txt> -k <kth_best> -a <alpha>'
			sys.exit()
		elif opt in ("-b", "--bow"):
			bowfile = arg
		elif opt in ("-c", "--cls"):
			clsfile = arg
		elif opt in ("-t", "--tst"):
			tstfile = arg
		elif opt in ("-T", "--tstcls"):
			tstclsfile = arg
		elif opt in ("-k", "--k"):
			k = int(arg)
		elif opt in ("-a", "--alpha"):
			alpha = float(arg)

    # Get bag of words array and sentiment array
	bow = read_bagofwords_dat(bowfile)
	cls = read_class_values_dat(clsfile)
	tst = read_bagofwords_dat(tstfile)
	tstcls = read_class_values_dat(tstclsfile)

	# Standard model
	model = BernoulliNB()
	model.fit(bow, cls)

	# Predict test set
	predict = model.predict_proba(tst)
	classes = model.predict(tst)

	# Write to file
	write_probs_to_file("./BNBbi_standard_probs.txt", predict)
	write_classes_to_file("./BNBbi_standard_classes.txt", classes)

	# KBest model
	sel = SelectKBest(f_classif, k)
	sel.fit(bow, cls)
	bowmod = sel.transform(bow)
	tstmod = sel.transform(tst)
	varmodel = BernoulliNB()
	varmodel.fit(bowmod, cls)

	varpredict = varmodel.predict_proba(tstmod)
	varclasses = varmodel.predict(tstmod)

	# Write to file
	write_probs_to_file("./BNBbi_kbest_probs.txt", varpredict)
	write_classes_to_file("./BNBbi_kbest_classes.txt", varclasses)

	# FPR model
	sel = SelectFpr(f_classif, alpha)
	sel.fit(bow, cls)
	bowmod = sel.transform(bow)
	tstmod = sel.transform(tst)
	varmodel = BernoulliNB()
	varmodel.fit(bowmod, cls)
	varpredict = varmodel.predict_proba(tstmod)
	varclasses = varmodel.predict(tstmod)

	# Write to file
	write_probs_to_file("./BNBbi_fpr_probs.txt", varpredict)
	write_classes_to_file("./BNBbi_fpr_classes.txt", varclasses)

	# Runtime
	print 'Runtime:', str(time.time() - start_time)

if __name__ == "__main__":
  main(sys.argv[1:])

