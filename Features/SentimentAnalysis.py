
from pycorenlp import *
import numpy as np

nlp=StanfordCoreNLP("http://192.168.1.26:9000/")


def SentiStr2Val( s ):
	if s == 'Negative':
		return -1
	if s == 'Positive':
		return 1
	return 0

def corenlpSentiment( text ):
	output = nlp.annotate(text, properties={"annotators":"sentiment",  "outputFormat": "json","triple.strict":"true" ,  'timeout': 30000 })
	sentences =  output['sentences']
	sentimentScores = [  int(sen['sentimentValue'])*SentiStr2Val( sen['sentiment']  ) for sen in sentences ]

	# print [  sen['sentiment'] for sen in sentences  ]

	return np.mean( sentimentScores )



# corenlpSentiment("i am a bad boy")