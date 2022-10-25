import os
import glob
import argparse
import colorama
import cv2 as cv
import numpy as np

from utils import progress_bar

def geometric_transform(src_file, dst_file):
	src_img = cv.imread(src_file)

	# origin picture points
	pts1 = np.float32([[627.2307, 166.6923], [4161.8461, 185.9230],
                       [350.3076, 3616.6923], [4442.6153, 3620.5384]])
	# new picture points
	pts2 = np.float32([[0, 0], [1024, 0], [0, 1024], [1024, 1024]])

	# Generate transform matrix
	M = cv.getPerspectiveTransform(pts1, pts2)
	# GeometricTransformations
	out = cv.warpPerspective(src_img, M, (1024, 1024))

	cv.imwrite(dst_file, out)

def create_calibration_picture(src_path, dst_path):
	src_img = glob.glob(src_path + "*.jpg")
	total = len(src_img)
	
	# generate pictures by geometric transform
	for i, image in enumerate(src_img):
		progress_bar(i, total - 1, color=colorama.Fore.YELLOW)
		geometric_transform(image, dst_path + image.split('/')[-1])
	print("\r\n")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--src", help = "source folder path")
	parser.add_argument("--dst", help = "destination of folder path")
	args = parser.parse_args()

	if not args.src:
		raise AssertionError("Error: Please input correct folder path")

	if not os.path.exists(args.dst):
		print("Create output folder")
		os.makedirs(args.dst)

	print("Input dir = ", args.src)
	print("Output dir = ", args.dst)
	print("Start geometric transform!!")
	create_calibration_picture(args.src, args.dst)
	print("complete!")