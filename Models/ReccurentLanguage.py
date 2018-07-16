from keras.layers import *
from keras.models import *


def simpleGRU( nVocab ):


	model = Sequential()
	model.add(Embedding(nVocab, 150,  input_length=30 ))
	model.add(GRU( 300  , return_sequences=False ))

	# model.load_weights("data/converted2.h5")

	model.add(Dropout(0.2))
	model.add(Dense( 2 ))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])

	return model
