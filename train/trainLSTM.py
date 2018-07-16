

import json
import sys ; sys.path.insert(0, '..')
import numpy as np

import Models , ProcessData , Datasets , Config 

nVocab = ProcessData.Tweets.nVocab
model = Models.ReccurentLanguage.simpleGRU( nVocab )

def getGen( cvsFiles=[] , splitPercentage=0.2 , split='train' , downsampeling=False ):

	positives , negatives = Datasets.SensitiveTweets.getTweets(  cvsFiles ,  splitPercentage , split , downsampeling  , out='pairs' )


	def getSentences():
		while True:
			p = positives.pop(0)
			n = negatives.pop(0)
			positives.append(p)
			negatives.append(n)

			yield p , n

	def gen():
		batch_size = 32
		G = getSentences()
		while True:
			X = []
			Y = []
			XI = []
			
			for i in range(batch_size):
				p , n = G.next()


				X.append( ProcessData.Tweets.getSentenceVec( ProcessData.TweetParser.anonomizeTweet(p['text']) ))
				X.append( ProcessData.Tweets.getSentenceVec( ProcessData.TweetParser.anonomizeTweet( n['text']) ))

				XI.append( ProcessData.Images.getImageVec(p['img']  , width=224 , height=224 ) )
				XI.append( ProcessData.Images.getImageVec(n['img']  , width=224 , height=224 ) )

				Y.append(  np.array([0,1]) )
				Y.append(  np.array([1,0]) )

			yield   np.array(X) , np.array( Y )

	return gen()

import glob

files = glob.glob("../../data/raw/sensi_annotated_tweets_1/*.csv")


v = getGen(cvsFiles=files , downsampeling=False   )
v2 = getGen(  cvsFiles=files , split='val' , downsampeling=False  )

print model.evaluate_generator( v , 5000  )
print model.evaluate_generator( v2 , 5000  )
print "o"

model.fit_generator( v , samples_per_epoch=500 , validation_data=v2  , nb_val_samples=200,  nb_epoch=100 )






