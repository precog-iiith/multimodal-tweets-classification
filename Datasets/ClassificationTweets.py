import json
import random

import Config

project_root = Config.project_root



def getBinClassTweetsJsons( class0=[] ,  class1=[] , splitPercentage=0.2 , split='train' , split_level='tweets' , downsampeling=False  ):

	if split_level == 'files':
		if split == 'train':
			class0 = class0[: int(  (1-splitPercentage)*len(class0)  ) ]
			class1 = class1[: int(  (1-splitPercentage)*len(class1)  ) ]
		else:
			class0 = class0[int(  (1-splitPercentage)*len(class0)  )  : ]
			class1 = class1[int(  (1-splitPercentage)*len(class1)  )  : ]


	class0_tw = []
	class1_tw = []

	class0 = sorted( class0 )
	class1 = sorted( class1 )

	for fname in class0:
		class0_tw += json.loads( open(fname).read()  )
	
	for fname in class1:
		class1_tw += json.loads( open(fname).read()  )

	for t in class0_tw:
		t['label'] = 0

	for t in class1_tw:
		t['label'] = 1

	random.seed( 100 )
	random.shuffle( class0_tw )

	random.seed( 100 )
	random.shuffle( class1_tw )

	if downsampeling:
		minL = min(len(class0_tw)  , len(class1_tw) )
		class0_tw = class0_tw[:minL]
		class1_tw = class1_tw[:minL]
	
	
	alll = class0_tw + class1_tw
	random.seed( 100 )
	random.shuffle( alll )

	if split_level == 'tweets':
		if split == 'train':
			alll = alll[: int(  (1-splitPercentage)*len(alll)  ) ]
		else:
			alll = alll[int(  (1-splitPercentage)*len(alll)  )  : ]

	return alll



