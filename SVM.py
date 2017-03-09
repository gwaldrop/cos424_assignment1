import numpy
import scipy
import sys
import time
import getopt
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
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
		print 'Usage: \n python SVM.py -b <bagofwords_csv> -c <classes_txt> -t <tst_bagofwords_csv>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'Usage: \n python SVM.py -b <bagofwords_csv> -c <classes_txt> -t <tst_bagofwords_csv>'
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

	# Binarize bagofwords representations
	for x in bow:
		for y in x:
			if y > 1:
				y = 1

	for x in tst:
		for y in x:
			if y > 1:
				y = 1

	# Train models
	svc = LinearSVC(loss='hinge', dual=True)
	svc.fit(bow, cls)

	# Predict test set
	predict = svc.decision_function(tst)
	classes = svc.predict(tst)

	# Compare and log results
	given = numpy.loadtxt("./tst_classes_1.txt");
	count = 0
	for (x, y) in numpy.nditer([given, classes], [], [['readonly'], ['readonly']]):
		if x == y:
			count += 1

	logfile=open("./rbfsvm_log.txt", 'w')
	logfile.write('LinearSVM w/o Feature Selection\n')
	logfile.write('Correctly predicted: ' + str(count) + ' of 600\n')
	logfile.write('Percent: ' + str(count/600.0) + '\n')

	# Write classes and probabilities
	outfile= open("./tst_probs_LinearSVM.txt", 'w')
	numpy.savetxt(outfile, predict, '%.4f', ', ', '\n')
	outfile.close()

	outfile= open("./tst_classes_LinearSVM.txt", 'w')
	numpy.savetxt(outfile, classes, '%d', ', ', '\n')
	outfile.close()

	svc = SVC(kernel='rbf')
	svc.fit(bow, cls)

	# Predict test set
	predict = svc.decision_function(tst)
	classes = svc.predict(tst)

	# Compare and log results
	count = 0
	for (x, y) in numpy.nditer([given, classes], [], [['readonly'], ['readonly']]):
		if x == y:
			count += 1

	logfile.write('\nRBFSVM w/o Feature Selection\n')
	logfile.write('Correctly predicted: ' + str(count) + ' of 600\n')
	logfile.write('Percent: ' + str(count/600.0) + '\n')

	# Write classes and probabilities
	outfile= open("./tst_probs_RBFSVM.txt", 'w')
	numpy.savetxt(outfile, predict, '%.4f', ', ', '\n')
	outfile.close()

	outfile= open("./tst_classes_RBFSVM.txt", 'w')
	numpy.savetxt(outfile, classes, '%d', ', ', '\n')
	outfile.close()

	# Perform feature selection using k best features
	logfile.write('\nRBF SVM w/ K Best Features\n')
	for i in range(1, 12):
		sel = SelectKBest(f_classif, i*100)
		sel.fit(bow, cls)
		bowmod = sel.transform(bow)
		tstmod = sel.transform(tst)
		varmodel = SVC(kernel='rbf')
		varmodel.fit(bowmod, cls)

		varclasses = varmodel.predict(tstmod)

		count = 0
		for (x, y) in numpy.nditer([given, varclasses], [], [['readonly'], ['readonly']]):
			if x == y:
				count += 1
		logfile.write('\nw/' + str(i*100) + '\n')
		logfile.write('Correctly predicted: ' + str(count) + ' of 600\n')
		logfile.write('Percent: ' + str(count/600.0) + '\n')

	# Feature selection using False Positive Rate Test
	logfile.write('\nRBF SVM w/ FPR Feature Selection\n')
	for i in range(1, 15):
		sel = SelectFpr(f_classif, i * 0.02)
		sel.fit(bow, cls)
		bowmod = sel.transform(bow)
		tstmod = sel.transform(tst)
		varmodel = SVC(kernel='rbf')
		varmodel.fit(bowmod, cls)
		varclasses = varmodel.predict(tstmod)

		count = 0
		for (x, y) in numpy.nditer([given, varclasses], [], [['readonly'], ['readonly']]):
			if x == y:
				count += 1
		logfile.write('\nw/ alpha = ' + str(i * 0.02) + '\n')
		logfile.write('Correctly predicted: ' + str(count) + ' of 600\n')
		logfile.write('Percent: ' + str(count/600.0) + '\n')

	logfile.close()

	# Runtime
	print 'Runtime:', str(time.time() - start_time)

if __name__ == "__main__":
  main(sys.argv[1:])