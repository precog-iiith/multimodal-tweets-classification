
# simple utility to manage cache nd stuff
# only works with np arrays currently

import numpy as np
import os.path
import hashlib
import pickle
import inspect
import json
import base64



import os

project_root = os.path.dirname( os.path.abspath(__file__ + "/../../") )


# Note that first elemnt of params shoule be the name

def getCacheFname(  params , cacheVersion=""  ):
	params = [ str(p) for p in params ]
	cacheHash = "|".join(params) + "|" + cacheVersion
	cacheHash = hashlib.md5(cacheHash).hexdigest()
	cacheFname = project_root+ "/data/cache/"+ params[0] + "-" +  cacheHash+".npy"
	return cacheFname

def isFCached(  params , cacheVersion=""  ):
	return os.path.isfile(getCacheFname(params , cacheVersion))

def saveFCache( params , content , cacheVersion="" , usePickle=False ):

	if usePickle:
		with open( getCacheFname(params , cacheVersion)  , 'wb') as handle:
			pickle.dump(content, handle)
	else:
		np.save( getCacheFname(params , cacheVersion) , content  )

	with open(project_root + "/data/cache/CachedParamsList-" + str(params[0]) + ".b64list"  , "a") as myfile:
		params_str = [ str(p) for p in params ]
		myfile.write( base64.b64encode(json.dumps(params_str)) + "\n")


def getFCached(  params , cacheVersion="" , usePickle=False ):

	if usePickle:
		with open( getCacheFname(params , cacheVersion)   , 'rb') as handle:
			return pickle.load(handle)
	else:
		return np.load(getCacheFname(params , cacheVersion) )


# http://stackoverflow.com/questions/5929107/python-decorators-with-parameters
def parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer


#todo reletive paths maybe 
@parametrized
def Cachify( fn , cacheVersion="" , ignoreArgs=[] , fileHashArgs=[] , ignoreFnName=False ):


	fnName = fn.__name__
	paramsNames =  inspect.getargspec(fn)[0]
	# defaults = list(inspect.getargspec(fn)[-1])

	def wrapper(*args, **kwargs):

		paramsDict = {}
		for i , pv in enumerate( args ):
			paramsDict[ paramsNames[i] ] = pv
		for k in kwargs:
			paramsDict[k ] = kwargs[k]

		for ia in ignoreArgs:
			del paramsDict[ia] 

		paramItems = paramsDict.items()
		paramItems.sort( key=lambda xx:xx[0] )
		cacheAtters = [ str(xx[1]) for xx in paramItems ]
		cacheAtters = [ str(fnName)+ str(cacheVersion) ] + cacheAtters

		if isFCached( cacheAtters ):
			return getFCached( cacheAtters , usePickle=True )


		# if not found check for file hash 
		fileHashArgsDict = {}
		for ar in paramsDict:
			fileHashArgsDict[ar] = paramsDict[ar]
		for fha in fileHashArgs:
			fileHashArgsDict[ fha ] = getFileHash( paramsDict[fha]  )
		paramItemsFH = fileHashArgsDict.items()
		paramItemsFH.sort( key=lambda xx:xx[0] )
		cacheAttersFH = [ str(xx[1]) for xx in paramItemsFH ]
		cacheAttersFH = [ "FileHash" + str(fnName)+ str(cacheVersion) ] + cacheAttersFH

		if isFCached( cacheAttersFH ):
			ccc = getFCached( cacheAttersFH , usePickle=True )
			saveFCache( cacheAtters , ccc  , usePickle=True  )
			return ccc

		ccc = fn(*args, **kwargs)
		saveFCache( cacheAtters , ccc  , usePickle=True  )
		saveFCache( cacheAttersFH , ccc  , usePickle=True  )

		return ccc


	def isCached(*args, **kwargs):
		paramsDict = {}
		for i , pv in enumerate( args ):
			paramsDict[ paramsNames[i] ] = pv
		for k in kwargs:
			paramsDict[k ] = kwargs[k]

		for ia in ignoreArgs:
			del paramsDict[ia] 

		paramItems = paramsDict.items()
		paramItems.sort( key=lambda xx:xx[0] )
		cacheAtters = [ str(xx[1]) for xx in paramItems ]
		cacheAtters = [ str(fnName)+ str(cacheVersion) ] + cacheAtters

		if isFCached( cacheAtters ):
			return True

		fileHashArgsDict = {}
		for ar in paramsDict:
			fileHashArgsDict[ar] = paramsDict[ar]
		for fha in fileHashArgs:
			fileHashArgsDict[ fha ] = getFileHash( paramsDict[fha]  )
		paramItemsFH = fileHashArgsDict.items()
		paramItemsFH.sort( key=lambda xx:xx[0] )
		cacheAttersFH = [ str(xx[1]) for xx in paramItemsFH ]
		cacheAttersFH = [ "FileHash" + str(fnName)+ str(cacheVersion) ] + cacheAttersFH

		if isFCached( cacheAttersFH ):
			return True
			
		return False

	wrapper.isCached = isCached 

	return wrapper



def getFileHash( fName ):
	return hashlib.md5(open(fName , 'rb').read()).hexdigest()


# print the list of params that are cached
def printAllCached( firstParam ):
	fName = project_root+"/data/cache/CachedParamsList-" + str(firstParam) + ".b64list" 
	with open( fName ) as f:
		for line in f:
			print base64.b64decode( line )


def getAllCached( firstParam ):
	fName = project_root+"/data/cache/CachedParamsList-" + str(firstParam) + ".b64list" 
	with open( fName ) as f:
		for line in f:
			yield json.loads(base64.b64decode( line ))


