
from Cache import Cachify

import numpy as np 
import hashlib
import sys

from PretrainedKeras.resnet50 import ResNet50
from PretrainedKeras.vgg16 import VGG16
from PretrainedKeras.vgg19 import VGG19
from PretrainedKeras.inception_v3 import InceptionV3
from PretrainedKeras.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.models import Model



@Cachify(cacheVersion="0.0" , fileHashArgs=['img_path'] )
def VGG16Features( img_path , intermedLayer=None ):

	# eg intermedLayer =  block4_pool 
	
	try:
		base_model = VGG16Features.BaseModel
	except AttributeError:
		base_model = VGG16Features.BaseModel = VGG16(weights='imagenet' )

	if intermedLayer is None:
		model = base_model
	else:

		try:
			___ = VGG16Features.interModels
		except AttributeError:
			VGG16Features.interModels = {}

		if not str(intermedLayer) in VGG16Features.interModels :
			VGG16Features.interModels[str(intermedLayer)] = Model(input=base_model.input, output=base_model.get_layer(intermedLayer).output)

		model = VGG16Features.interModels[str(intermedLayer)]

	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	vggVecs = model.predict(x)


	return vggVecs






@Cachify(cacheVersion="0.0" , fileHashArgs=['img_path'] )
def VGG19Features( img_path , intermedLayer=None ):

	# eg intermedLayer =  block4_pool 
	
	try:
		base_model = VGG19Features.BaseModel
	except AttributeError:
		base_model = VGG19Features.BaseModel = VGG19(weights='imagenet' )

	if intermedLayer is None:
		model = base_model
	else:

		try:
			___ = VGG19Features.interModels
		except AttributeError:
			VGG19Features.interModels = {}

		if not str(intermedLayer) in VGG19Features.interModels :
			VGG19Features.interModels[str(intermedLayer)] = Model(input=base_model.input, output=base_model.get_layer(intermedLayer).output)

		model = VGG19Features.interModels[str(intermedLayer)]

	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	vggVecs = model.predict(x)

			

	return vggVecs




@Cachify(cacheVersion="0.0" , fileHashArgs=['img_path'] )
def ResNetFeatures( img_path , intermedLayer=None ):

	# eg intermedLayer =  block4_pool 
	
	try:
		base_model = ResNetFeatures.BaseModel
	except AttributeError:
		base_model = ResNetFeatures.BaseModel = ResNet50(weights='imagenet' )

	if intermedLayer is None:
		model = base_model
	else:

		try:
			___ = ResNetFeatures.interModels
		except AttributeError:
			ResNetFeatures.interModels = {}

		if not str(intermedLayer) in ResNetFeatures.interModels :
			ResNetFeatures.interModels[str(intermedLayer)] = Model(input=base_model.input, output=base_model.get_layer(intermedLayer).output)

		model = ResNetFeatures.interModels[str(intermedLayer)]

	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	vggVecs = model.predict(x)

		

	return vggVecs




@Cachify(cacheVersion="0.0" , fileHashArgs=['img_path'] )
def InceptionV3Features( img_path , intermedLayer=None ):

	# eg intermedLayer =  block4_pool 
	
	try:
		base_model = InceptionV3Features.BaseModel
	except AttributeError:
		base_model = InceptionV3Features.BaseModel = InceptionV3(weights='imagenet' )

	if intermedLayer is None:
		model = base_model
	else:

		try:
			___ = InceptionV3Features.interModels
		except AttributeError:
			InceptionV3Features.interModels = {}

		if not str(intermedLayer) in InceptionV3Features.interModels :
			InceptionV3Features.interModels[str(intermedLayer)] = Model(input=base_model.input, output=base_model.get_layer(intermedLayer).output)

		model = InceptionV3Features.interModels[str(intermedLayer)]

	img = image.load_img(img_path, target_size=(299, 299))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	vggVecs = model.predict(x)

	return vggVecs



@Cachify(cacheVersion="0.0" , fileHashArgs=['img_path'] )
def CaffeVGG16Features( img_path ) :
	from CaffeModels import Imagenet_VGG
	return Imagenet_VGG.getResult( img_path )



@Cachify(cacheVersion="0.0" , fileHashArgs=['img_path'] )
def CaffeMITPlacesVGG( img_path ) :
	from CaffeModels import MITPlaces_VGG
	return MITPlaces_VGG.getResult( img_path)



@Cachify(cacheVersion="0.0" , fileHashArgs=['img_path'] )
def CaffeMITPlacesGoogleNet( img_path ) :
	from CaffeModels import MITPlaces_googlenet
	return MITPlaces_googlenet.getResult( img_path)



@Cachify(cacheVersion="0.0" , fileHashArgs=['img_path'] )
def CaffeSOSVGG( img_path ) :
	from CaffeModels import SOS_VGG
	return SOS_VGG.getResult( img_path)



@Cachify(cacheVersion="0.0" , fileHashArgs=['img_path'] )
def CaffeSOSGoogleNet( img_path ) :
	from CaffeModels import SOS_Googlenet
	return SOS_Googlenet.getResult( img_path)


@Cachify(cacheVersion="0.0" , fileHashArgs=['img_path'] )
def RCNNHumans( img_path ):
	sys.path.insert(0, '/opt/py-faster-rcnn/tools')
	import faster_rcnn_api
	return faster_rcnn_api.getHumanBoxes( img_path )




# import CNNFeatures
# CNNFeatures.VGG16Features("/home/divam/RHS.jpg" , 'fc2')
# CNNFeatures.ResNetFeatures("/home/divam/LHS.jpg")
