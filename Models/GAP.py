


from keras.models import Sequential
from keras.layers import Reshape
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam , SGD
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.regularizers import ActivityRegularizer
from keras import backend as K
from keras.regularizers import activity_l1 , activity_l2 , activity_l1l2
from keras.layers import Input, Dense, Lambda
from VGG import getVGG16Model
from keras.utils.layer_utils import layer_from_config

from keras.models import Model




# vgg with global average pooling 
# nLeyarsSegerate how many layers end from retained layers, how many should have isolated instances

def VGG_GAP( nClasses , nLayersRetain = 31 , nFreez=0 , nLeyarsSegerate=0 ,  useWeights=True , optimizer=None , dropout=0.5 , weightsFile=None   ):

	from keras.layers.pooling import GlobalAveragePooling2D

	
	model = getVGG16Model( useWeights=useWeights , nFreez= nFreez , nClasses=None  )
	for i in range( len(model.layers) - nLayersRetain ):
		model.pop()
	

	model.add( GlobalAveragePooling2D() )

	if not dropout is None:
		model.add(Dropout(dropout))

	

	if not weightsFile is None:
		model.add( Dense( nOutputsInOldWeights ) )
		model.load_weights( weightsFile )
		model.pop() 

	model.add(Dense( nClasses ))
	model.add(Activation('softmax'))

	if not optimizer is None:
		model.compile(loss="categorical_crossentropy", optimizer= optimizer , metrics=['accuracy'] )

	return model



