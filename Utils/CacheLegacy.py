
# simple utility to manage cache nd stuff
# only works with np arrays currently

import numpy as np
import os.path
import hashlib
import pickle


def getCacheFname(  params , cacheVersion=""  ):
	params = [ str(p) for p in params ]
	cacheHash = "|".join(params) + "|" + cacheVersion
	cacheHash = hashlib.md5(cacheHash).hexdigest()
	cacheFname = "../data/cache_leg/"+cacheHash+".npy"
	return cacheFname

def isFCached(  params , cacheVersion=""  ):
	return os.path.isfile(getCacheFname(params , cacheVersion))

def saveFCache( params , content , cacheVersion="" , usePickle=False ):
	 
	if usePickle:
		with open( getCacheFname(params , cacheVersion)  , 'wb') as handle:
			pickle.dump(content, handle)
	else:
		np.save( getCacheFname(params , cacheVersion) , content  )

def getFCached(  params , cacheVersion="" , usePickle=False ):

	if usePickle:
		with open( getCacheFname(params , cacheVersion)   , 'rb') as handle:
			return pickle.load(handle)
	else:
		return np.load(getCacheFname(params , cacheVersion) )





