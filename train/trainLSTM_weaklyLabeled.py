

import json
import sys ; sys.path.insert(0, '..')
import numpy as np

import Models , ProcessData , Datasets , Config 

nVocab = ProcessData.Tweets.nVocab
model = Models.ReccurentLanguage.simpleGRU( nVocab )

batch_size = 32

def pointGen( data ):
	while True:
		for d in data:
			txt = ProcessData.TweetParser.anonomizeTweet( d['text'] )
			x = ProcessData.Tweets.getSentenceVec( txt )
			if d['label'] == 0:
				y = np.array([1,0])
			else:
				y = np.array([0,1])
			yield x , y 

def batchGen( data ):
	g = pointGen(data)
	while True:
		X = []
		Y = []
		for i in range(batch_size):
			x,y = g.next()
			X.append(x)
			Y.append(y)
		yield   np.array(X) , np.array( Y )




import glob


pos = glob.glob( "../../data/tweets/sensitive_hashtag_tweets/*.json" ) 
neg = glob.glob( "../../data/tweets/non_sensitive_hashtag_tweets/*.json" ) 



tr_da = Datasets.ClassificationTweets.getBinClassTweetsJsons( pos , neg , split='train' , downsampeling=True ,split_level='files' )
te_da = Datasets.ClassificationTweets.getBinClassTweetsJsons( pos , neg , split='test'  , downsampeling=True ,split_level='files' )


v = batchGen(tr_da )
v2 = batchGen(  te_da )




i2w=  ProcessData.Tweets.idx2word
import pickle
with open('/home/divamg/a/i2w.pkl', 'wb') as handle:
    pickle.dump( i2w , handle )



model.load_weights("../../model_weights/sensitive_vs_non_sensitive_bulk_ep40.h5")


while True:
	print "i"
	txt = raw_input()
	txt = ProcessData.TweetParser.anonomizeTweet( txt )
	x = ProcessData.Tweets.getSentenceVec( txt )
	x = np.array([x])
	print list(x)




