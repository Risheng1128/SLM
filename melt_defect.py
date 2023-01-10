import cv2 as cv
import numpy as np
import sys
import os
import glob
import argparse
import colorama
from utils import progress_bar

def rotate_img(img):
    (h, w, d) = img.shape
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, -90, 1.0)
    return cv.warpAffine(img, M, (w, h))

def struct_mask_image(mask_path):
    new_width = 4320
    new_height = 4320
    mask = cv.imread(mask_path)
    mask = rotate_img(mask)

    rows, cols = mask.shape[:2]
    mask = cv.resize(mask, (new_width, new_height))
    mask = cv.resize(mask, None, fx=1.46, fy=1.48, interpolation=cv.INTER_LINEAR)

    # shift image
    M = np.float32([[1, 0, 70], [0, 1, 200]])
    mask = cv.warpAffine(mask, M, (new_width, new_height), borderValue=(255, 255, 255))
    mask = mask[0:new_width, 0:new_height]

    return mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help = "source path")
    parser.add_argument("--mask", help = "mask path")
    parser.add_argument("--dst", help = "destination path")
    args = parser.parse_args()

    img_list = list(enumerate(glob.glob(args.src + "*.jpg")))
    img_list = sorted([i[1] for i in img_list])
    mask_list = list(enumerate(glob.glob(args.mask + "*.bmp")))
    mask_list = sorted([i[1] for i in mask_list])
    image_total = int(len(img_list))
    mask_total = int(len(mask_list))

    if (not image_total) or (not mask_total):
    	raise AssertionError("Error: no input image!")

    if not os.path.isdir(args.dst):
        os.makedirs(args.dst)

    print("Input image dir = ", args.src)
    print("Input mask dir = ", args.mask)
    print("Output dir = ", args.dst)

    for i in range(image_total):
        progress_bar(i, image_total - 1, color=colorama.Fore.YELLOW)
        result_path = args.dst + img_list[i].split('/')[-1] + "/"
        if not os.path.isdir(result_path):
            os.makedirs(result_path)

        real_img = cv.imread(img_list[i])
        real_mask = struct_mask_image(mask_list[i])

        cv.imwrite(result_path + "origin.jpg", real_img)
        cv.imwrite(result_path + mask_list[i].split('/')[-1], real_mask)
        cv.imwrite(result_path + "or.jpg", cv.bitwise_or(real_mask, real_img))
        cv.imwrite(result_path + "or_not_img.jpg", cv.bitwise_or(real_mask, cv.bitwise_not(real_img)))
        cv.imwrite(result_path + "and_not_mask.jpg", cv.bitwise_and(cv.bitwise_not(real_mask), real_img))
        cv.imwrite(result_path + "not_or.jpg", cv.bitwise_not(cv.bitwise_or(real_mask, real_img)))

    print("Complete!")