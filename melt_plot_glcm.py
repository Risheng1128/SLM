import argparse
import cv2 as cv
import os
import fast_glcm

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--src', help = 'source image path')
	parser.add_argument('--dst', help = 'destination of folder path')
	args = parser.parse_args()

	if not args.src:
	    raise AssertionError('Source path not found!!')

	if not os.path.isdir(args.dst):
		print('No directory. Construct one')
		os.makedirs(args.dst)

	img = cv.imread(args.src)
	img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	cv.imwrite(args.dst + 'origin.jpg', img)

	for i in [0, 45, 90, 135]:
		std = fast_glcm.fast_glcm_std(img, angle=i)
		asm, ene = fast_glcm.fast_glcm_ASM(img, angle=i)

		cv.imwrite(args.dst + str(i) + '_std.jpg', std)
		cv.imwrite(args.dst + str(i) + '_ene.jpg', ene)
