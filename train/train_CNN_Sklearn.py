
"""

this takes urls.txt amd trains a classifer with pre cached feature vectors


"""


import Download
import CNNFeatures
import numpy as np


from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB



def getFeature( url , useCached=True  ):

	if useCached and (not Download.isDownlaoded(url ) ) :
		return None

	fname = Download.Download( url )
	if useCached and ( not  CNNFeatures.CaffeVGG16Features.isCached( fname ) ):
		return None
	if useCached and ( not  CNNFeatures.CaffeMITPlacesVGG.isCached( fname ) ):
		return None

	return np.array( CNNFeatures.CaffeVGG16Features(fname)+CNNFeatures.CaffeMITPlacesVGG(fname)   )


def getFeatures( urls , maxCount = None  ):
	F = []
	for url in urls:
		f = getFeature( url )
		if not f is None:
			F.append( f )

		if (not maxCount is None) and len(F) >= maxCount:
			break

	return np.array( F )

positive_urls = open("../data/protests.txt").read()
positive_urls = positive_urls.split('\n')


negative_urls = open("../data/GDELT_URLS_50k.txt").read()
negative_urls = negative_urls.split('\n')




def evaluate_results( predicted , groundTruth  ):
	#print 'The precision for this classifier is ' + str(metrics.precision_score(groundTruth, predicted))
	print 'The recall for this classifier is ' + str(metrics.recall_score(groundTruth, predicted))
	print 'The f1 for this classifier is ' + str(metrics.f1_score(groundTruth, predicted))
	print 'The accuracy for this classifier is ' + str(metrics.accuracy_score(groundTruth, predicted))
	print '\nHere is the classification report:'
	print classification_report(groundTruth, predicted)
	print '\nHere is the confusion matrix:'
	print metrics.confusion_matrix(groundTruth, predicted)
	print "\nCross-validation scores: " 


X_positives = getFeatures( positive_urls , maxCount=15000  )
X_negatives = getFeatures( negative_urls , maxCount=15000  )


print "No of positive features : " ,  X_positives.shape[0]
print "No of negative features : " ,  X_negatives.shape[0]

X = np.concatenate((X_positives , X_negatives ))
Y = np.array( [0]*X_negatives.shape[0] + [1]*X_positives.shape[0]  )


model = linear_model.LogisticRegression(C=10 )


print cross_validation.cross_val_score( model , X, Y , cv=5 )

model.fit( X , Y )
predicted = model.predict( X )

evaluate_results( predicted , Y ) 


import cPickle
# save the classifier
with open('classifier_logistic_terror_1.pkl', 'wb') as fid:
	cPickle.dump(model, fid)    




