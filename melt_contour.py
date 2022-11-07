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

# move the item isolated from original image to the median of image
# default image size = 4320 * 4320
def move_item_median(img, contour, median_width=2160, median_height=2160):
    # get the center of contour
    M = cv.moments(contour)
    c_width = int(M["m10"]/M["m00"])
    c_height = int(M["m01"]/M["m00"])

    M = np.float32([[1, 0, median_width - c_width], [0, 1, median_height - c_height]])
    img = cv.warpAffine(img, M, (median_width * 2, median_height * 2))
    return img

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
        # read mask image
        origin_mask_img = cv.imread(mask_list[i], cv.COLOR_BGR2GRAY)
        # find all contours in mask
        contours = get_contour(origin_mask_img)

        # the absolutely path of mask folder
        mask_path_list = mask_list[i].split('/')
        mask_path = ""
        for i in mask_path_list[0:-1]:
            mask_path += i + '/'
        origin_img = cv.imread(mask_path + 'origin.jpg', cv.COLOR_BGR2GRAY)        

        # create output path
        output_path = args.dst + mask_path_list[-1] + '/'
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        cv.imwrite(output_path + 'mask.bmp', origin_mask_img)
        cv.imwrite(output_path + 'origin.jpg', origin_img)
        index = 0
        for c in contours:
            # eliminate contour that is too small 
            if cv.contourArea(c) < 20:
                continue

            mask = np.zeros(origin_mask_img.shape, dtype='uint8')
            cv.drawContours(mask, [c], -1, (255, 255, 255), -1)

            # separate workpieces image
            workpiece_img = cv.bitwise_and(origin_img, mask)
            cv.imwrite(output_path + 'workpiece_' + str(index) + '.jpg', workpiece_img)

            # move workpieces median
            median_img = move_item_median(workpiece_img, c)
            cv.imwrite(output_path + 'median_' + str(index) + '.jpg', median_img)

            # the mask image of separate workpieces
            workpiece_mask_img = cv.bitwise_or(origin_mask_img, cv.bitwise_not(mask))
            cv.imwrite(output_path + 'workpiece_mask_' + str(index) + '.jpg', workpiece_mask_img)

            # ~(workpiece + workpiece_mask)
            workpiece_not_or_img = cv.bitwise_not(cv.bitwise_or(workpiece_img, workpiece_mask_img))
            cv.imwrite(output_path + 'workpiece_not_or' + str(index) + '.jpg', workpiece_not_or_img)
            index += 1
