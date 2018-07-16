
import numpy as np

import TweetParser



vocab = open("../../data/other/vocab.txt").read().split("\n")
vocab = [ v.strip().lower() for v in vocab  ]
nVocab = len(vocab) - 1
print "nVocab " , nVocab


idx2word = {}
word2idx = {}

for i,w in enumerate(vocab):
	word2idx[w] = i
	idx2word[i] = w

# word2idx = json.loads( open("data/w2i.json").read()  )
# nVocab = len(word2idx)
# print "nVocab " , nVocab

def getSentenceVec( sentence ):
	words = sentence.split(' ')
	L = []
	for w in words:
		w = w.lower()
		if w in word2idx:
			if word2idx[w]  < nVocab:
				L.append( word2idx[w]  )

	if len(L) > 30 :
		L = L[:30]
	if len(L) < 30:
		L = [0]*(30-len(L)) + L
	return  np.array(L)



def getSentenceCharLevel( sentence ):
	X = np.zeros(( 150 , 38 ))
	chars = "qwertyuiopasdfghjklzxcvbnm1234567890 "
	sentence = sentence.lower()

	sentence = "".join([ ch for ch in sentence if ch in chars  ])

	for i,ch in enumerate( sentence[::-1] ) :
		if i >= 150:
			break
		j = chars.index( ch  )
		X[ 150 - 1 - i , j  ] = 1
		
	return X
		



