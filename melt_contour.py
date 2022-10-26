import cv2 as cv
import numpy as np
import argparse
import glob
import os
import colorama
from utils import progress_bar

# get the contours of image
def get_contour(img):
    ret, binary = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    blur_img = cv.GaussianBlur(binary, (3, 3), 0)
    canny_img = cv.Canny(blur_img, 80, 200)
    contours, _ = cv.findContours(canny_img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', help = 'source folder path')
    parser.add_argument('--dst', help = 'destination of folder path')
    args = parser.parse_args()

    if not args.src:
        raise AssertionError('Source path not found!!')

    if not os.path.isdir(args.dst):
        print('No directory. Construct one')
        os.makedirs(args.dst)

    mask_list = glob.glob(args.src + '**/*.bmp')
    list_num = len(mask_list)
    for i in range(list_num):
        progress_bar(i, list_num - 1, color=colorama.Fore.YELLOW)
        img = cv.imread(mask_list[i], cv.COLOR_BGR2GRAY)
        contours = get_contour(img)

        # create output path
        output_path = args.dst + mask_list[i].split('/')[-1] + '/'
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        cv.imwrite(output_path + 'origin.bmp', img)
        index = 0
        for c in contours:
            # eliminate contour that is too small 
            if cv.contourArea(c) < 20:
                continue

            mask = np.zeros(img.shape, dtype='uint8')
            cv.drawContours(mask, [c], -1, (255, 255, 255), -1)
            # cv.imwrite(args.dst + str(i) + '.jpg', cv.bitwise_and(test_img, mask))
            cv.imwrite(output_path + str(index) + '.jpg', mask)
            index += 1
