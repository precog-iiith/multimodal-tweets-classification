
import requests
from Cache import Cachify
import json


@Cachify(cacheVersion="0.1" , fileHashArgs=['img_path'] )
def FaceExpression( img_path ) :

	url = "https://mememoji.rhobota.com/v1.0.0/predict"

	r = requests.post(url, files={'image_buf': open(img_path, 'rb')})
	return json.loads( r.text )

	
# print FaceExpression( "a.jpg")

