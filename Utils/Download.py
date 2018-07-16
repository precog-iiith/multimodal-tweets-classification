

import hashlib
import urllib2
import os


def isDownlaoded( url , outputDir="../data/downloaded/"  ):
	fExt = "." + url.split(".")[-1]

	if len(fExt) > 5:
		fExt = ""

	fName = str(hashlib.md5(url).hexdigest()) + fExt
	fPath = outputDir + fName

	if os.path.isfile(  fPath ):
		return True
	return False

# returns the path of the downloaded file
def Download( url , outputDir="../data/downloaded/" , userAgent="Googlebot-Image/1.0" , ext=None ) :

	fExt = "." + url.split(".")[-1]

	if len(fExt) > 5:
		fExt = ""

	fName = str(hashlib.md5(url).hexdigest()) + fExt
	fPath = outputDir + fName

	if not ext is None:
		fPath = fPath + ext

	if os.path.isfile(  fPath ):
		return fPath

	opener = urllib2.build_opener()
	opener.addheaders = [('User-Agent',  userAgent )]
	downloadFile = opener.open( url , timeout=5 )

	with open(  fPath  ,'wb') as output:
		output.write(downloadFile.read())

	return fPath



if __name__ == "__main__":

	print Download("http://static.dnaindia.com/sites/default/files/styles/third/public/2016/07/18/483090-unesco-heritage-list.jpg")
	
