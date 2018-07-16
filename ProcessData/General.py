
import numpy as np

def getClassificationVector( classId , nClasses ):
	v = np.zeros( nClasses )
	v[classId] = 1
	return v

def getMultiClassificationVector( classIds , nClasses ):
	v = np.zeros( nClasses )
	for classId in classIds:
		v[classId] = 1
	return v

def bindSingleToMultiVecFn( singleVecFn  ):

	def fn( dataObjects , *args, **kwargs  ):
		X = []
		Y = []

		for d in dataObjects:
			x , y = singleVecFn( d ,  *args, **kwargs )
			X.append( x )
			Y.append( y )

		return np.array(X ) , np.array( Y )

	return fn


# this is for functions returning multiple  data points of diffrent count
def bindSingleToMultiVecFn2( singleVecFn , verbose=False ):

	def fn( dataObjects , *args, **kwargs  ):
		X = None
		Y = None

		if verbose:
			import progressbar
			bar = progressbar.ProgressBar()
			II = bar(dataObjects)
		else:
			II = dataObjects

		for d in II:
			x , y = singleVecFn( d ,  *args, **kwargs )
			if X is None:
				X = x 
				Y = y 
			else:
				X = np.concatenate((X , x ))
				Y = np.concatenate((Y , y ))
			
			

		return X , Y

	return fn




def bindSingleFusionToMultiVecFn( singleVecFn  , nXTerms ):

	def fn( dataObjects , *args, **kwargs  ):
		XX = [[] for i in range(nXTerms)]
		Y = []

		for d in dataObjects:

			x , y = singleVecFn( d ,  *args, **kwargs )

			for i in range(nXTerms):
				XX[i].append( x[i] )
			Y.append( y )

		XX = [ np.array(X ) for X in XX ]

		return XX , np.array( Y )

	return fn



def getBindedFn( fn ,  *args, **kwargs ):
	def wrapped( dataObjects , baseImagesPath="" ):
		return fn(dataObjects , baseImagesPath , *args , **kwargs )
	return wrapped



