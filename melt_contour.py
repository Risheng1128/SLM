import cv2 as cv
import numpy as np
import argparse
import glob
import os
import colorama
from utils import progress_bar

def rotate_img(img):
    (h, w, d) = img.shape
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, -90, 1.0)
    return cv.warpAffine(img, M, (w, h))

def create_mask_image(mask_path):
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

# crop the image
def crop_image(img, x=1648, y=1648, width=1024, height=1024):
    return img[y:y+height, x:x+width]

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
    parser.add_argument('--src', help = 'source path')
    parser.add_argument("--mask", help = "mask path")
    parser.add_argument('--dst', help = 'destination path')
    args = parser.parse_args()

    if (not args.src) or (not args.mask) or (not args.dst):
        raise AssertionError('Error: Input path is not found')

    if not os.path.isdir(args.dst):
        os.makedirs(args.dst)

    origin_list = list(enumerate(glob.glob(args.src + '*.jpg')))
    origin_list = sorted([i[1] for i in origin_list])

    mask_list = list(enumerate(glob.glob(args.mask + '*.bmp')))
    mask_list = sorted([i[1] for i in mask_list])
    mask_num = len(mask_list)

    # layer
    for i in range(mask_num):
        progress_bar(i, mask_num - 1, color=colorama.Fore.YELLOW)

        origin_img = cv.imread(origin_list[i], cv.COLOR_BGR2GRAY)        
        mask_img = create_mask_image(mask_list[i])

        # find all contours in mask (item number)
        contours = get_contour(mask_img)

        # workpiece
        for c in range(len(contours)):
            # eliminate contour that is too small 
            if cv.contourArea(contours[c]) < 20:
                continue

            # create output path
            output_path = args.dst + 'item' + str(c + 1) + '/'
            if not os.path.isdir(output_path):
                os.makedirs(output_path)

            mask = np.zeros(mask_img.shape, dtype='uint8')
            cv.drawContours(mask, [contours[c]], -1, (255, 255, 255), -1)

            # separate workpieces image
            mask = cv.bitwise_and(mask, cv.bitwise_not(mask_img))
            workpiece_img = cv.bitwise_and(origin_img, mask)
            workpiece_img = move_item_median(workpiece_img, contours[c])
            workpiece_img = crop_image(workpiece_img)

            if i < 9:
                cv.imwrite(output_path + 'layer_0' + str(i + 1) + '.jpg', workpiece_img)
            else:
                cv.imwrite(output_path + 'layer_' + str(i + 1) + '.jpg', workpiece_img)
