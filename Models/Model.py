from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers.core import Flatten
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import UpSampling2D , MaxPooling2D 
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten
from keras.optimizers import Adam
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.layers.noise import GaussianNoise
import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.regularizers import ActivityRegularizer
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras import backend as K

from parse import parse # pip install parse
from let import let # pip i let

import Vectorize






class UnPooling2D(Layer):
    """A 2D Repeat layer"""
    def __init__(self, poolsize=(2, 2)):
        super(UnPooling2D, self).__init__()
        self.poolsize = poolsize

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], input_shape[1],
                self.poolsize[0] * input_shape[2],
                self.poolsize[1] * input_shape[3])

    def get_output(self, train):
        X = self.get_input(train)
        s1 = self.poolsize[0]
        s2 = self.poolsize[1]
        output = X.repeat(s1, axis=2).repeat(s2, axis=3)
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
            "poolsize":self.poolsize}






def segnetModel(nClasses):

	kernel = 3
	filter_size = 64
	pad = 1
	pool_size = 2

	autoencoder = models.Sequential()
	autoencoder.add(Layer(input_shape=(3, Vectorize.IMG_HIEGHT, Vectorize.IMG_WIDTH)))

	# encoder
	autoencoder.add(ZeroPadding2D(padding=(pad,pad)))
	autoencoder.add(Convolution2D(filter_size, kernel, kernel, border_mode='valid'))
	autoencoder.add(BatchNormalization())
	autoencoder.add(Activation('relu'))
	autoencoder.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

	autoencoder.add(ZeroPadding2D(padding=(pad,pad)))
	autoencoder.add(Convolution2D(128, kernel, kernel, border_mode='valid'))
	autoencoder.add(BatchNormalization())
	autoencoder.add(Activation('relu'))
	autoencoder.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

	autoencoder.add(ZeroPadding2D(padding=(pad,pad)))
	autoencoder.add(Convolution2D(256, kernel, kernel, border_mode='valid'))
	autoencoder.add(BatchNormalization())
	autoencoder.add(Activation('relu'))
	autoencoder.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

	autoencoder.add(ZeroPadding2D(padding=(pad,pad)))
	autoencoder.add(Convolution2D(512, kernel, kernel, border_mode='valid'))
	autoencoder.add(BatchNormalization())
	autoencoder.add(Activation('relu'))


	# decoder
	autoencoder.add( ZeroPadding2D(padding=(pad,pad)))
	autoencoder.add( Convolution2D(512, kernel, kernel, border_mode='valid'))
	autoencoder.add( BatchNormalization())

	autoencoder.add( UpSampling2D(size=(pool_size,pool_size)))
	autoencoder.add( ZeroPadding2D(padding=(pad,pad)))
	autoencoder.add( Convolution2D(256, kernel, kernel, border_mode='valid'))
	autoencoder.add( BatchNormalization())

	autoencoder.add( UpSampling2D(size=(pool_size,pool_size)))
	autoencoder.add( ZeroPadding2D(padding=(pad,pad)))
	autoencoder.add( Convolution2D(128, kernel, kernel, border_mode='valid'))
	autoencoder.add( BatchNormalization())

	autoencoder.add( UpSampling2D(size=(pool_size,pool_size)))
	autoencoder.add( ZeroPadding2D(padding=(pad,pad)))
	autoencoder.add( Convolution2D(filter_size, kernel, kernel, border_mode='valid'))
	autoencoder.add( BatchNormalization())


	autoencoder.add(Convolution2D( nClasses , 1, 1, border_mode='valid',))
	autoencoder.add(Reshape(( nClasses ,Vectorize.IMG_HIEGHT*Vectorize.IMG_WIDTH), input_shape=( nClasses ,Vectorize.IMG_HIEGHT,Vectorize.IMG_WIDTH)))
	autoencoder.add(Permute((2, 1)))
	autoencoder.add(Activation('softmax'))
	#from keras.optimizers import SGD
	#optimizer = SGD(lr=0.01, momentum=0.8, decay=0., nesterov=False)
	
	return autoencoder





# vgg with global average pooling 
def VGG_GAP( nClasses , nLayersRetain = 23, nFreez=0 , useWeights=True  ):

	from keras.layers.pooling import GlobalAveragePooling2D

	model = getVGG16Model( useWeights=useWeights , nFreez= nFreez , nClasses=None  )

	for i in range( len(model.layers) - nLayersRetain ):
		model.pop()

	model.add( GlobalAveragePooling2D() )
	model.add(Dense( nClasses ))
	model.add(Activation('softmax'))

	return model








def getVGG16Model( useWeights=True , nFreez = 0 , nClasses=None  ):
	model = Sequential()
	 
	model.add(ZeroPadding2D((1,1),input_shape=(  3, Vectorize.VGGIMG_SIZE , Vectorize.VGGIMG_SIZE  )))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1000, activation='softmax'))

	if useWeights == True:
		model.load_weights('../data/vgg16_weights.h5')

	if not ((nClasses is None ) or (nClasses == 1000 )  ) :
		model.layers.pop()
		model.add(Dropout(0.5))
		model.add(Dense( nClasses ,  activation='softmax' ))

	freezeLayers = model.layers[: nFreez]

	for l in freezeLayers:
		l.trainable = False

	return model




def loadWeights( weightsPath ):
	with h5py.File(weightsPath) as hw:
        for k in range(hw.attrs['nb_layers']):
            g = hw['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
            if model.layers[k].name == "convolution2d_13":
                break
        print('Model loaded.')

        




def getModel(modelName):
	

	
	if let( u = parse("segnet_{ch}classes" , modelName )):
		m = segnetModel(int(u['ch']))
		m.compile(loss="categorical_crossentropy", optimizer='adadelta')

	elif  let( u =  parse("vgg16_{ch}classes_{nf}freeze" , modelName )):
		m = getVGG16Model( useWeights=True , nFreez = int(u['nf']) , nClasses=int(u['ch']))
		m.compile(loss="categorical_crossentropy", optimizer='adadelta' , metrics=['accuracy'])

	else:
		raise Exception("Invelid Model Name")


	

	return m

if __name__ == "__main__":
	from keras.utils.visualize_util import plot
	m = segnetModel(2)
	# plot(m, to_file='../data/model.png' , show_shapes=True)







