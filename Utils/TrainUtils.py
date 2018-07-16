
import LoadData
import math
import Vectorize
import Cache
from threading import Thread
import numpy as np

class VectorizerWorker:

	def __init__( self , modelName , dataPath , batchSize=300 , dontCache=True , parallelJobs = True , nWorkers=5  , batchesDone=0) :
		
		self.modelName = modelName
		self.dataPath = dataPath
		self.batchSize = batchSize
		self.dontCache = dontCache
		self.parallelJobs = parallelJobs
		self.nWorkers = nWorkers
		self.vecCache = {}
		self.baseImagePath = "/".join(dataPath.split("/")[:-1]) + "/"
		self.data = LoadData.loadJSONData( dataPath )
		self.totalBatches =  int( math.ceil( ( self.data.length  +0.0) / batchSize ) )
		self.useFileCaching = False # for now
		self.batchesDone = batchesDone # n of batches done already
		self.isStopped = False
		self.totalBatchesDone = 0
		self.totalDataPoints = self.data.length

		if parallelJobs:
			self.parallelVecCacherJob()


	def vectorizeNCacheInBg(self , batchNo):
		# this ill asyncronously vectorize and cache a given batch

		batchNo = int( int(batchNo)%self.totalBatches) # so that it doesnt overflow
		if batchNo in self.vecCache:
			return

		if self.useFileCaching:
			cacheAttrs = [ 'vecCache-05' , batchNo , self.batchSize , self.modelName ]
			if  Cache.isFCached(cacheAttrs) :
				self.vecCache[batchNo] = Cache.getFCached(cacheAttrs, usePickle=True) 
				return

		def doCache(batchNo):
			# print "Starting vectorisation on battch " + str(batchNo)
			batch = self.data.get( batchNo*self.batchSize ,  (batchNo+1)*self.batchSize )
			self.vecCache[batchNo] = Vectorize.getVector( self.modelName , batch , self.baseImagePath  )

			if self.useFileCaching:
				cacheAttrs = [ 'vecCache-05' , batchNo , self.batchSize , self.modelName ]
				Cache.saveFCache(cacheAttrs , self.vecCache[batchNo] , usePickle=True)
			# print "Done vectorisation on battch "  + str(batchNo)

		t = Thread(target=doCache, args=(batchNo,) )
		t.start()



	def parallelVecCacherJob( self ):

		def workerFn():

			for i in range(self.nWorkers):
				self.vectorizeNCacheInBg(self.batchesDone +i)

			batchesDonePrev = self.batchesDone 

			while True:
				while batchesDonePrev == self.batchesDone  :

					if self.isStopped:
						break

				if self.isStopped:
					break
					
				batchesDonePrev = self.batchesDone
				self.vectorizeNCacheInBg( self.batchesDone + self.nWorkers )

		t = Thread(target=workerFn, args=() )
		t.setDaemon(True )
		t.start()



	def getVecs(self ,  batchNo ):

		if not batchNo in self.vecCache:
			batch = self.data.get( batchNo*self.batchSize  ,  (batchNo+1)*self.batchSize  )
			self.vecCache[batchNo] = Vectorize.getVector( self.modelName  , batch , self.baseImagePath  )

		vecs =  self.vecCache[batchNo]

		if self.dontCache:
			self.vecCache.pop(batchNo)

		return vecs

	def getNext( self ): # get the next set of vectors

		v =  self.getVecs(  self.batchesDone )
		self.batchesDone = (  self.batchesDone  + 1 )%self.totalBatches
		self.totalBatchesDone += 1 
		return v

	def generator( self ): # returns a generatr which will will be passed in keras fit_generator
		while True:
			yield self.getNext()

	def reset( self ):
		self.batchesDone  = 0

	def stop( self ):
		self.isStopped = True

	def __exit__( self ):
		self.stop()



class stateManager:

	def __init__( self , expname ) :
		self.expname = expname

	def saveState():
		pass

	def attachModel( model ):
		pass

	def attachVectoriser( vectoriser ):
		pass

	def loadState():
		pass







def trainModel( model , train_vectorizer , nEpochs , val_vectoriser = None , minibatchsize = 30 ):

	while True:
		vec = train_vectorizer.getNext()
		cur_epoch =  train_vectorizer.totalBatchesDone% train_vectorizer.totalBatches
		cur_batch = train_vectorizer.batchesDone

		if cur_epoch >= nEpochs:
			break

		print "Epoch " , cur_epoch , "Batch " , cur_batch
		if val_vectoriser is None:
			loss =  model.fit( *vec ,  batch_size=minibatchsize , nb_epoch=1 ).history['loss'][-1]
			print "Loss -> " , loss
		else :
			loss =  model.fit( *vec ,  batch_size=minibatchsize , nb_epoch=1 , validation_data= val_vectoriser.getNext()   ).history['loss'][-1]
			print "Loss -> " , loss
		



def trainModel2( model ,  train_vectorizer , nBatchIter ):
	
	import progressbar
	bar = progressbar.ProgressBar()

	losses = []

	for i in bar(range(nBatchIter)):
		vec = train_vectorizer.getNext()
		loss = model.train_on_batch( *vec  )

		losses.append( loss )

	print "Loss -> " , np.mean( losses )



# returns ground tructh array and corresponding predicted array
def getGroundTruthNPredictions( model , vectorizer  ):

	# vectorizer.reset()
	# predictions = model.predict_generator( vectorizer.generator() ,  vectorizer.totalDataPoints )

	Y = None
	YP = None

	vectorizer.reset()
	for i in range( vectorizer.totalBatches ):

		x , y  = vectorizer.getNext()
		yp = model.predict_on_batch( x )

		if Y is None:
			Y = y
		else:
			Y = np.concatenate( ( Y , y ))

		if YP is None:
			YP = yp
		else:
			YP = np.concatenate( ( YP , yp ))

	return Y , YP











# from TrainUtils import trainModel2 , VectorizerWorker
# import Model
# m = Model.getModel('vggsegnet_2classes')
# v = VectorizerWorker( 'vggsegnet_2classes' , "../data/prepped/egohands/train.json" , 12 )
# m.fit_generator( v.generator() , samples_per_epoch=v.totalDataPoints , nb_epoch=10 , show_accuracy=True )


# trainModel2( m , v , 100 )

# 3 21 
# m2.fit( *vec ,  batch_size=12 , nb_epoch=1 )
# 3:08
# v = VectorizerWorker( 'vggsegnet_2classes' , "../data/prepped/egohands/t.json" , 30 )
