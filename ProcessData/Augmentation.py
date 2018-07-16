
# this module will create augnentations of video frames / image frames

import numpy as np
import cv2
import random
import time
import math
import glob
import os






# http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders

def rotate_image(image, angle):
	"""
	Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
	(in degrees). The returned image will be large enough to hold the entire
	new image, with a black background
	"""

	# Get the image size
	# No that's not an error - NumPy stores image matricies backwards
	image_size = (image.shape[1], image.shape[0])
	image_center = tuple(np.array(image_size) / 2)

	# Convert the OpenCV 3x2 rotation matrix to 3x3
	rot_mat = np.vstack(
		[cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
	)

	rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

	# Shorthand for below calcs
	image_w2 = image_size[0] * 0.5
	image_h2 = image_size[1] * 0.5

	# Obtain the rotated coordinates of the image corners
	rotated_coords = [
		(np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
		(np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
		(np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
		(np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
	]

	# Find the size of the new image
	x_coords = [pt[0] for pt in rotated_coords]
	x_pos = [x for x in x_coords if x > 0]
	x_neg = [x for x in x_coords if x < 0]

	y_coords = [pt[1] for pt in rotated_coords]
	y_pos = [y for y in y_coords if y > 0]
	y_neg = [y for y in y_coords if y < 0]

	right_bound = max(x_pos)
	left_bound = min(x_neg)
	top_bound = max(y_pos)
	bot_bound = min(y_neg)

	new_w = int(abs(right_bound - left_bound))
	new_h = int(abs(top_bound - bot_bound))

	# We require a translation matrix to keep the image centred
	trans_mat = np.matrix([
		[1, 0, int(new_w * 0.5 - image_w2)],
		[0, 1, int(new_h * 0.5 - image_h2)],
		[0, 0, 1]
	])

	# Compute the tranform for the combined rotation and translation
	affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

	# Apply the transform
	result = cv2.warpAffine(
		image,
		affine_mat,
		(new_w, new_h),
		flags=cv2.INTER_LINEAR
	)

	return result


def largest_rotated_rect(w, h, angle):
	"""
	Given a rectangle of size wxh that has been rotated by 'angle' (in
	radians), computes the width and height of the largest possible
	axis-aligned rectangle within the rotated rectangle.

	Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

	Converted to Python by Aaron Snoswell
	"""

	quadrant = int(math.floor(angle / (math.pi / 2))) & 3
	sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
	alpha = (sign_alpha % math.pi + math.pi) % math.pi

	bb_w = w * math.cos(alpha) + h * math.sin(alpha)
	bb_h = w * math.sin(alpha) + h * math.cos(alpha)

	gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

	delta = math.pi - alpha - gamma

	length = h if (w < h) else w

	d = length * math.cos(alpha)
	a = d * math.sin(alpha) / math.sin(delta)

	y = a * math.cos(gamma)
	x = y * math.tan(gamma)

	return (
		bb_w - 2 * x,
		bb_h - 2 * y
	)


def crop_around_center(image, width, height):
	"""
	Given a NumPy / OpenCV 2 image, crops it to the given width and height,
	around it's centre point
	"""

	image_size = (image.shape[1], image.shape[0])
	image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

	if(width > image_size[0]):
		width = image_size[0]

	if(height > image_size[1]):
		height = image_size[1]

	x1 = int(image_center[0] - width * 0.5)
	x2 = int(image_center[0] + width * 0.5)
	y1 = int(image_center[1] - height * 0.5)
	y2 = int(image_center[1] + height * 0.5)

	return image[y1:y2, x1:x2]






def augment( frame_paths , outputDir , baseImagesPath="" , seed=None   ):

	imgNo = 0
	allOuts = []

	cv_images = [  cv2.imread( baseImagesPath + frame, 1) for frame in frame_paths ]

	augmentedImages = augmentCVImages(cv_images , sed=seed )

	for dst in augmentedImages:
		imgNo += 1
		outFName = str(imgNo).zfill(5) + ".png"
		cv2.imwrite( outputDir +  outFName, dst  )

		allOuts.append( outFName )
	return allOuts




def augmentCVImage(  cv_image , seed=None  ):
	return augmentCVImages( [cv_image] , seed=seed )[0]


def augmentCVImages( cv_images , seed=None  ):

	local_random = random.Random()
	
	local_random.seed( seed )

	augmentedImages = []
	flip = local_random.randint(0,1)

	zoom = local_random.randint( 75 , 100 ) + 0.0
	widthStretch = local_random.randint(85 , 105 ) + 0.0

	rotation = local_random.randint(-10 , 10 ) + 0.0
	rotationZ = local_random.randint( -7 , 6 ) + 0.0

	shiftXpercent = local_random.randint(-100 , 100)
	shiftYpercent = local_random.randint(-100 , 100)

	r1 = local_random.randint(95 , 105)/100.0
	r2 = local_random.randint(95 , 105)/100.0
	r3 = local_random.randint(95 , 105)/100.0
	r4 = local_random.randint(95 , 105)/100.0
	r5 = local_random.randint(95 , 105)/100.0
	r6 = local_random.randint(95 , 105)/100.0
	r7 = local_random.randint(95 , 105)/100.0
	r8 = local_random.randint(95 , 105)/100.0
	

	for im in cv_images:

		img = np.copy(im)

		height, width = img.shape[:2]

		image_orig = np.copy(img)
		image_rotated = rotate_image(img, rotation )
		image_rotated_cropped = crop_around_center(
			image_rotated,
			*largest_rotated_rect(
				width,
				height,
				math.radians( rotation )
			)
		)


		img = image_rotated_cropped
		height, width = img.shape[:2]

		windowX1 = windowX3 = max( width/2 - (width * (zoom/100)*(widthStretch/100) )/2 , 0)
		windowX2 = windowX4 = min(width/2 + (width * (zoom/100)*(widthStretch/100) )/2 , width )
		windowY1 = windowY2 = max( height/2 - (height * (zoom/100) )/2 , 0 )
		windowY3 = windowY4 = min(height/2 + (height * (zoom/100) )/2 , height )

		# print windowX1

		shiftX = windowX1*shiftXpercent/100
		shiftY = windowY1*shiftYpercent/100

		windowX1 -= shiftX
		windowX2 -= shiftX
		windowX3 -= shiftX
		windowX4 -= shiftX
		windowY1 -= shiftY
		windowY2 -= shiftY
		windowY3 -= shiftY
		windowY4 -= shiftY

		if flip == 1:
			windowX1 , windowX2 , windowX3 , windowX4 = windowX2 , windowX1 , windowX4 , windowX3

		windowX1 = windowX1*r1
		windowX2 = windowX2*r2
		windowX3 = windowX3*r3
		windowX4 = windowX4*r4
		windowY1 = windowY1*r5
		windowY2 = windowY2*r6
		windowY3 = windowY3*r7
		windowY4 = windowY4*r8

		# print "appapa" , windowY1 , windowX1

		windowX1 = int(windowX1)
		windowX2 = int(windowX2)
		windowX3 = int(windowX3)
		windowX4 = int(windowX4)
		windowY1 = int(windowY1)
		windowY2 = int(windowY2)
		windowY3 = int(windowY3)
		windowY4 = int(windowY4)


		pts1 = np.float32([ [windowX1 , windowY1] , [windowX2, windowY2] , [windowX3 , windowY3] , [windowX4 , windowY4] ])
		pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
		

		M = cv2.getPerspectiveTransform(pts1,pts2)
		dst = cv2.warpPerspective(img,M,(width , height ))


		augmentedImages.append(  np.copy(dst)  )

		# cv2.imshow('dst_rt', dst )
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		# print "appapa" , windowY1 , windowX1


	return augmentedImages




if __name__ == "__main__":
	# demo()
	d = glob.glob("../data/tmp/frames/1/*.png")
	d.sort()
	d = d[:20]
	i = 0

	seed = random.randint(0 , 1000 )

	for dd in d:
		im = cv2.imread(dd , 1)
		im2 = augmentCVImage( im , seed )
		# i += 100
		cv2.imshow('dst_rt', im2 )
		cv2.waitKey(0)
		# cv2.destroyAllWindows()

	# dirr = "../data/tmp/xyz/"

	# if not os.path.exists(dirr):
	# 	os.makedirs(dirr)

	# print augment( d , dirr    )

	


			


	
