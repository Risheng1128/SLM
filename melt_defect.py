import cv2 as cv
import numpy as np
import sys
import os
import glob
import argparse

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
    parser.add_argument("--src", help = "source folder path")
    parser.add_argument("--mask", help = "mask folder path")
    parser.add_argument("--dst", help = "destination of folder path")
    args = parser.parse_args()

    img_list = list(enumerate(glob.glob(args.src + "*.jpg")))
    img_list = sorted([i[1] for i in img_list])
    mask_list = list(enumerate(glob.glob(args.mask + "*.bmp")))
    mask_list = sorted([i[1] for i in mask_list])
    image_total = int(len(img_list))
    mask_total = int(len(mask_list))

    if (not image_total) or (not mask_total):
    	raise AssertionError("No input image!!")

    if not os.path.isdir(args.dst):
        print("No directory. Construct one")
        os.makedirs(args.dst)

    print("Input image dir = ", args.src)
    print("Input mask dir = ", args.mask)
    print("Output dir = ", args.dst)

    for i in range(image_total):
        result_path = args.dst + img_list[i].split('/')[-1] + "/"
        if not os.path.isdir(result_path):
            os.makedirs(result_path)

        real_img = cv.imread(img_list[i])
        real_mask = struct_mask_image(mask_list[i])

        cv.imwrite(result_path + "origin.jpg", real_img)
        cv.imwrite(result_path + mask_list[i].split('/')[-1], real_mask)
        cv.imwrite(result_path + "or.jpg", cv.bitwise_or(real_mask, real_img))
        cv.imwrite(result_path + "and.jpg", cv.bitwise_and(real_mask, real_img))
        cv.imwrite(result_path + "xor.jpg", cv.bitwise_xor(real_mask, real_img))
        cv.imwrite(result_path + "not_and.jpg", cv.bitwise_not(cv.bitwise_and(real_mask, real_img)))
        cv.imwrite(result_path + "and_not_img.jpg", cv.bitwise_and(real_mask, cv.bitwise_not(real_img)))
        cv.imwrite(result_path + "or_not_img.jpg", cv.bitwise_or(real_mask, cv.bitwise_not(real_img)))
        cv.imwrite(result_path + "and_not_mask.jpg", cv.bitwise_and(cv.bitwise_not(real_mask), real_img))
        cv.imwrite(result_path + "or_not_mask.jpg", cv.bitwise_or(cv.bitwise_not(real_mask), real_img))
        cv.imwrite(result_path + "xor_not.jpg", cv.bitwise_xor(cv.bitwise_not(real_mask), real_img))

        # Create binary image
        not_or_img = cv.bitwise_not(cv.bitwise_or(real_mask, real_img))
        cv.imwrite(result_path + "not_or.jpg", not_or_img)
        ret, not_or_img = cv.threshold(not_or_img, 135, 255, cv.THRESH_BINARY)
        cv.imwrite(result_path + "not_or_binary.jpg", not_or_img)

    print("Complete!")