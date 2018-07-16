
import numpy as np
import cv2

import glob
import os
import random
import General


IMG_WIDTH = 480
IMG_HIEGHT = 360

VGGIMG_SIZE = 224

# crop coordinates -> [[x1,y1] , [x2,y2] ] 
def getImageVec(path , width=IMG_WIDTH , height=IMG_HIEGHT , cropCoordinates=None ,   augment=False , augmentationSeed = None , imgNorm="sub_mean" , pipelineOrder=['crop' , 'augment' , 'resize' , 'norm'] , maintainAspectRatio=False ):

	if path == "":
		img = np.zeros((  height , width  , 3 ))
		img = np.rollaxis(img, 2, 0)
		return img

	try:
		img = cv2.imread(path, 1)

		for fn in pipelineOrder:
			
			if fn == 'crop':
				if  not cropCoordinates is None:
					[[x1,y1] , [x2,y2] ] = cropCoordinates
					[[x1,y1] , [x2,y2] ]  = [[int(x1),int(y1)] , [int(x2),int(y2)] ]  
					pts1 = np.float32([ [x1 , y1 ] , [x2, y1] , [x1 , y2] , [x2 , y2] ])
					pts2 = np.float32([[0,0],[abs(x2-x1),0],[0,abs(y2-y1)],[abs(x2-x1),abs(y2-y1)]])
					M = cv2.getPerspectiveTransform(pts1,pts2)
					img = cv2.warpPerspective(img,M,(abs(x2-x1) , abs(y2-y1)  ))
			
			if fn == 'augment':
				if augment:
					import Augmentation
					img = Augmentation.augmentCVImage( img , augmentationSeed )

			if fn == 'resize':
				if maintainAspectRatio:
					img_container = np.zeros((  height , width  , 3 ))
					h = np.size(img, 0)
					w = np.size(img, 1)
					ratio = (w+0.0)/h
					c1 = ( width ,  int( (width+0.0)/ratio   ) )
					c2 = ( int(height*ratio) , height   )
					if (c1[0] <= width and c1[1] <= height):
						img = cv2.resize(img, c1 )
					else:
						img = cv2.resize(img, c2 )
					img_container[0:0+img.shape[0], 0:0+img.shape[1]] = img
					img = img_container
				else:
					img = cv2.resize(img, ( width , height ))

			if fn == 'norm':
				if imgNorm == "sub_and_divide":
					img = np.float32( img ) / 127.5 - 1
				elif imgNorm == "sub_mean":
					img = img.astype(np.float32)
					img[:,:,0] -= 103.939
					img[:,:,1] -= 116.779
					img[:,:,2] -= 123.68
				elif imgNorm == "divide":
					img = img.astype(np.float32)
					img = img/255.0

		img = np.rollaxis(img, 2, 0)
		return img
	except Exception, e:
		# print path , e
		# print 
		img = np.zeros((  height , width  , 3 ))
		img = np.rollaxis(img, 2, 0)
		return img





def getImageClassificationVec( d , baseImagePath , nClasses , width=IMG_WIDTH , height=IMG_HIEGHT , augment=False , augmentationSeed = None , imgNorm="sub_mean" , maintainAspectRatio=False ):

	return (getImageVec( baseImagePath + d['img'] , width=width , height=height , augment=augment , augmentationSeed=augmentationSeed , imgNorm=imgNorm , maintainAspectRatio=maintainAspectRatio ) 
		,General.getClassificationVector(d['classId'] , nClasses ) )



def getImageMulticlassClassificationVec( d , baseImagePath , nClasses , width=IMG_WIDTH , height=IMG_HIEGHT , augment=False , augmentationSeed = None , imgNorm="sub_mean" , maintainAspectRatio=False ):

	return (getImageVec( baseImagePath + d['img'] , width=width , height=height , augment=augment , augmentationSeed=augmentationSeed , imgNorm=imgNorm , maintainAspectRatio=maintainAspectRatio ) 
		,General.getMultiClassificationVector(d['classIds'] , nClasses ) )






"""
this takes in a directory
directory/
	class2/
		img1.png
		img2.png
	class3/
		img1.png
		img2.png

"""
def getImageClassifierFromDirGen( imagesDir ,   width=IMG_WIDTH, height=IMG_HIEGHT , batchSize=32 ,  augment=False , split=None ):

	allImages = glob.glob( os.path.join( imagesDir , "*/*.jpg" )   ) + glob.glob( os.path.join( imagesDir , "*/*.jpeg" )   ) + glob.glob( os.path.join( imagesDir , "*/*.png" )   )
	random.seed( 1000 )
	random.shuffle( allImages )

	if not split is None:
		if split == 'test':
			allImages = allImages[:2000 ]
		else:
			allImages = allImages[ 2000: ]

	classes = set()
	for image in allImages :
		className = image.split('/')[-2]
		classes.add( className )

	classes = list(classes)
	classes = sorted( classes )
	nClasses = len(classes )
	classesIds = dict( enumerate(classes ))
	classesIds = {v: k for k, v in classesIds.iteritems()}

	X_batch = []
	Y_batch = []
	while True : 

		for image in allImages :

			X_batch.append( getImageVec(image , width=width , height=height , augment=augment )  )
			className = image.split('/')[-2]
			classId = classesIds[className]

			classVec = np.zeros(( nClasses ) )
			classVec[classId] = 1

			Y_batch.append( classVec )

			if len( X_batch ) == batchSize:
				tx , ty = X_batch , Y_batch
				X_batch = []
				Y_batch = []

				yield np.array( tx ) , np.array(ty)





