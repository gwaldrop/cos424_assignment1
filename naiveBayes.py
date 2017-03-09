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

def main(argv):
	
	start_time = time.time()
	print "running main()"

	bowfile = ''
	clsfile = ''
	tstfile = ''

	# Parse arguments
	try:
		opts, args = getopt.getopt(argv,"b:c:t:",["bow=","cls=","tst="])
	except getopt.GetoptError:
		print 'Usage: \n python naiveBayes.py -b <bagofwords_csv> -c <classes_txt> -t <tst_bagofwords_csv>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'Usage: \n python naiveBayes.py -b <bagofwords_csv> -c <classes_txt> -t <tst_bagofwords_csv>'
			sys.exit()
		elif opt in ("-b", "--bow"):
			bowfile = arg
		elif opt in ("-c", "--cls"):
			clsfile = arg
		elif opt in ("-t", "--tst"):
			tstfile = arg

    # Get bag of words array and sentiment array
	bow = read_bagofwords_dat(bowfile)
	cls = read_class_values_dat(clsfile)
	tst = read_bagofwords_dat(tstfile)

    # Train Naive Bayes model
	clfmodel = BernoulliNB()
	clfmodel.fit(bow, cls)

    # Predict test set
	predict = clfmodel.predict_proba(tst)
	classes = clfmodel.predict(tst)

	# Compare predicted classes to given classes in test set
	given = numpy.loadtxt("./tst_classes_1.txt");
	count = 0
	for (x, y) in numpy.nditer([given, classes], [], [['readonly'], ['readonly']]):
		if x == y:
			count += 1
	logfile=open("./tst_log.txt", 'w')
	logfile.write('Bernoulli Naive Bayes w/o Feature Selection\n')
	logfile.write('Correctly predicted: ' + str(count) + ' of 600\n')
	logfile.write('Percent: ' + str(count/600.0) + '\n')

    # Write classes and probabilities
	outfile= open("./tst_probs_NB.txt", 'w')
	numpy.savetxt(outfile, predict, '%.4f', ', ', '\n')
	outfile.close()

	outfile= open("./tst_classes_NB.txt", 'w')
	numpy.savetxt(outfile, classes, '%d', ', ', '\n')
	outfile.close()

	# Perform feature selection using k best features
	logfile.write('\nBernoulli Naive Bayes w/ K Best Features\n')
	for i in range(1, 12):
		sel = SelectKBest(f_classif, i*100)
		sel.fit(bow, cls)
		bowmod = sel.transform(bow)
		tstmod = sel.transform(tst)
		varmodel = BernoulliNB()
		varmodel.fit(bowmod, cls)

		varpredict = varmodel.predict_proba(tstmod)
		varclasses = varmodel.predict(tstmod)

		count = 0
		for (x, y) in numpy.nditer([given, varclasses], [], [['readonly'], ['readonly']]):
			if x == y:
				count += 1
		logfile.write('\nw/' + str(i*100) + '\n')
		logfile.write('Correctly predicted: ' + str(count) + ' of 600\n')
		logfile.write('Percent: ' + str(count/600.0) + '\n')

	# Feature selection using False Positive Rate Test
	logfile.write('\nBernoulli Naive Bayes w/ FPR Feature Selection\n')
	for i in range(1, 15):
		sel = SelectFpr(f_classif, i * 0.02)
		sel.fit(bow, cls)
		bowmod = sel.transform(bow)
		tstmod = sel.transform(tst)
		varmodel = BernoulliNB()
		varmodel.fit(bowmod, cls)
		varpredict = varmodel.predict_proba(tstmod)
		varclasses = varmodel.predict(tstmod)

		count = 0
		for (x, y) in numpy.nditer([given, varclasses], [], [['readonly'], ['readonly']]):
			if x == y:
				count += 1
		logfile.write('\nw/ alpha = ' + str(i * 0.02) + '\n')
		logfile.write('Correctly predicted: ' + str(count) + ' of 600\n')
		logfile.write('Percent: ' + str(count/600.0) + '\n')

	logfile.close()

	# Write classes and probabilities
	outfile= open("./tst_probs_VarNB.txt", 'w')
	numpy.savetxt(outfile, varpredict, '%.4f', ', ', '\n')
	outfile.close()

	outfile= open("./tst_classes_VarNB.txt", 'w')
	numpy.savetxt(outfile, varclasses, '%d', ', ', '\n')
	outfile.close()

  	# Runtime
	print 'Runtime:', str(time.time() - start_time)

if __name__ == "__main__":
  main(sys.argv[1:])
