




import csv
import json
import os
import random
import Config

project_root = Config.project_root

def getTweets( cvsFiles=[] , splitPercentage=0.2 , split='train' , downsampeling=False , attribute='label' , out='combined'):

	positives = []
	negatives = []

	for fname in cvsFiles :

		with open( fname ) as f:

			if '.csv' in fname:
				reader = csv.DictReader(f)
				reader = list( reader )
			if '.json' in fname:
				reader =  json.loads( f.read()  )
				
				
			print fname
			
			# print len( reader )

			if split == 'train':
				reader = reader[: int(  (1-splitPercentage)*len(reader)  ) ]
			else:
				reader = reader[int(  (1-splitPercentage)*len(reader)  )  : ]

			for row in reader:
				row['label'] = int( row['label'] )
				# print row['hatespeech']
				row['label'] = row[attribute ]

				if 'img_fn' in row and  len( row[ 'img_fn' ]) > 2:
					row['img'] =  project_root + "/data/prepped/tweets_annotated_2/images/" +  row[ 'img_fn' ].split("/")[-1]
				else:
					row['img']  = ""
				
				if int(row['label'])  != 1 :
					positives.append( row  )
				else:
					negatives.append( row )
					
			print len(positives ) , len( negatives)

			if downsampeling:
				minL = min(len(positives)  , len(negatives) )
				positives = positives[:minL]
				negatives = negatives[:minL]

	if out == 'combined':
		alll = positives + negatives
		random.shuffle( alll )
		return alll
	if out == 'pairs':
		return positives , negatives






