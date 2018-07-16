# -*- coding: utf-8 -*-


import re
import json
from collections import Counter

import spacy
nlp = None

URL_REGEX = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""

def tokeniseTweet(tweetTxt):
	
	tweetTxt = tweetTxt.replace("RT " , "").replace("RT:" , "")

	for p in ".,?!;":
		tweetTxt  = tweetTxt.replace(p+"#" , p+" #")
		tweetTxt  = tweetTxt.replace(p+"@" , p+" @")

	hashTags = re.finditer(r'\S*#(?:\[[^\]]+\]|\S+)', tweetTxt) # find hashtags
	mentions = re.finditer(r'\S*@(?:\[[^\]]+\]|\S+)', tweetTxt) # find mentions
	urls = re.finditer( URL_REGEX  , tweetTxt) # find urls
	emoticons =  re.finditer( r"([:=;][oO\-]?[D\)\]\(\]/\\OpPd])" , tweetTxt)  # find emoticons
	properWords = re.finditer("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",tweetTxt)
	#todo # remove "....." and shit
	# todo remove ascii art
	punctuations = re.finditer("[.,\/#!\?$%\^&\*;:{}=\-_`~()]",tweetTxt)


	# hashTags = re.findall(r'\S*#(?:\[[^\]]+\]|\S+)', tweetTxt) # find hashtags
	# mentions = re.findall(r'\S*@(?:\[[^\]]+\]|\S+)', tweetTxt) # find mentions
	# urls = re.findall( URL_REGEX  , tweetTxt) # find urls
	# emoticons =  re.findall( r"([:=;][oO\-]?[D\)\]\(\]/\\OpPd])" , tweetTxt)  # find emoticons
	# properWords = re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",tweetTxt)
	# #todo # remove "....." and shit
	# # todo remove ascii art
	# punctuations = re.findall("[.,\/#!\?$%\^&\*;:{}=\-_`~()]",tweetTxt)



	allEntityList = [ ('url',urls) , ('hashtag' , hashTags) , ('mention' , mentions) , ('emoji' , emoticons)  , ('word' , properWords), ('punctuation' , punctuations ) ]

	partsOfString = ["0"]*len(tweetTxt)
	allEntitities = {0 : "none" }

	
	for (entityType , entities ) in allEntityList:
		for p in entities:
			if partsOfString[ p.span()[0] : p.span()[1] ] == ['0']*(p.span()[1] - p.span()[0]): # if all zero
				i = len(allEntitities)
				allEntitities[i] = [entityType ,tweetTxt[p.span()[0] : p.span()[1]] ]
				partsOfString[ p.span()[0] : p.span()[1] ] = [ i ]*(p.span()[1] - p.span()[0])

	tokens = []

	partsOfString = ['0'] + partsOfString

	for i in range(1 , len(partsOfString )):
		if partsOfString[i] != partsOfString[i-1] and partsOfString[i] != '0':
			tokens.append(allEntitities[partsOfString[i]])

	tokens =  [  {"type":w[0] , "word":w[1]} for w in tokens ]
	return tokens


def anonomizeTweet(text ):
	global nlp
	if nlp is None:
		nlp = spacy.load('en')
	# python -m spacy.en.download all
	text = " ".join( [ t['word'] for t in tokeniseTweet( text) if t['type'] in ['word' , 'punctuation'] ] )
	doc = nlp( unicode(text) )
	txtRet = ""
	for d in doc:
		if d.ent_type_ != '':
			txtRet += "<%s> "%( d.ent_type_ )
		else:
			txtRet += d.text + " "
	return txtRet






	
if __name__ == "__main__":
	print  anonomizeTweet(" Another nonsense from miss nonsense. \n\n It is clear u do not know anything about India. Give up ur passport and move #Gandhi #baniya")


