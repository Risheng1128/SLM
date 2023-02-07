import os
import glob
import argparse
import colorama
import cv2 as cv
import numpy as np

from utils import progress_bar

def geometric_transform(src_file, dst_file):
	new_width = 4320
	new_height = 4320
	src_img = cv.imread(src_file)

	# origin picture points, get from Data/Geometric/origin.json
	pts1 = np.float32([[627.2307, 166.6923], [4161.8461, 185.9230],
                       [350.3076, 3616.6923], [4442.6153, 3620.5384]])
	# new picture points
	pts2 = np.float32([[0, 0], [new_width, 0], [0, new_height], [new_width, new_height]])

	# Generate transform matrix
	M = cv.getPerspectiveTransform(pts1, pts2)
	# GeometricTransformations
	out = cv.warpPerspective(src_img, M, (new_width, new_height))

	cv.imwrite(dst_file, out)

def create_calibration_picture(src_path, dst_path):
	src_img = glob.glob(src_path + "*.jpg")
	total = len(src_img)
	
	# generate pictures by geometric transform
	for i, image in enumerate(src_img):
		progress_bar(i, total - 1, color=colorama.Fore.YELLOW)
		geometric_transform(image, dst_path + image.split('/')[-1])

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--src',
	                    default='./data/melt/',
			    help='source image path')
	parser.add_argument('--dst',
	                    default='./result/geometric/',
	                    help='destination path')
	args = parser.parse_args()

	if not args.src:
		raise AssertionError("Error: source path not found!")

	if not os.path.exists(args.dst):
		os.makedirs(args.dst)

	create_calibration_picture(args.src, args.dst)