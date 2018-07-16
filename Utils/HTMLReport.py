import base64
import cv2
import unidecode

class HTMLReport():
	"""docstring for HTMLReport"""

	def __init__(self ):
		self.htmlBody = ""

	def addHead( self , txt , h="h1" ):
		self.htmlBody += "<%s> %s </%s>  " % ( h , txt , h )

	def addHR( self ):
		self.htmlBody += "<hr> "

	def addBR( self ):
		self.htmlBody += "<br> "

	def addImg( self , imgPath , resizeTo=None):
		img = cv2.imread(imgPath)
		if not resizeTo is None:
			img = cv2.resize(img , resizeTo )

		cnt = cv2.imencode('.jpg',img)[1]
		b64 = base64.encodestring(cnt)

		self.htmlBody += "<img src='data:image/jpg;base64,"+b64 +"'> <br>"

	def addCVImg( self , img , resizeTo=None):

		if not resizeTo is None:
			img = cv2.resize(img , resizeTo )

		cnt = cv2.imencode('.jpg',img)[1]
		b64 = base64.encodestring(cnt)

		self.htmlBody += "<img src='data:image/jpg;base64,"+b64 +"'> <br>"

	def addText(self , txt ):
		txt = unidecode.unidecode( txt )
		txt = str(txt)
		txt = txt.split("\n")
		txt = "<pre>" + "</pre> <br>\n<pre>".join(txt)   + "</pre> <br>\n"
		self.htmlBody += txt 

	def addHTML(self , txt ):
		self.htmlBody += txt


	def save(self , fname ):
		completeHTML = "<html> <body> " + self.htmlBody + "</html> </body> " 
		open(fname , 'wb').write( completeHTML )

