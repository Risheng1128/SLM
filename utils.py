import numpy as np
import matplotlib.pyplot as plt
import random
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import math
import colorama
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data.datasets import register_coco_instances

from detectron2.config import get_cfg
from detectron2 import model_zoo

def DICE_COEFFICIENT(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        print("NO INSTANCE!")
        return empty_score

    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

def dice_folder(ground_truth, prediction):
    pred = glob.glob(prediction)
    dice = []
    for i, image in enumerate(pred):
        folder = image.split('/')[-1].split('.')[0]
        gt = ground_truth + str(folder) + '/label.png'
        progress_bar(i, len(pred), color=colorama.Fore.YELLOW)

        try:
            GT = cv2.imread(gt)
            GT = cv2.cvtColor(GT, cv2.COLOR_BGR2GRAY)
            GT = cv2.threshold(GT, 1, 255, cv2.THRESH_BINARY)

            pred = cv2.imread(image)
            pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
            pred = cv2.threshold(pred, 1, 255, cv2.THRESH_BINARY)

            # print(gt, DICE_COEFFICIENT(pred[1], GT[1]))
            dice.append(DICE_COEFFICIENT(pred[1], GT[1]))

        except:
            pass

    return sum(dice) / len(dice)

def plot_samples(dataset_name, n=1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)

    for s in random.sample(dataset_custom, n):
        img = cv2.imread(s["file_name"])
        v = Visualizer(img[:, :, ::-1], metadata=dataset_custom_metadata, scale=0.5)
        v = v.draw_dataset_dict(s)
        plt.figure(figsize=(20, 20))
        plt.imshow(v.get_image())
        plt.show()

def regist_dataset(src_path):
    ## Regist Dataset
    register_coco_instances(name="Example", metadata={},
                            json_file=src_path + "annotations.json",
                            image_root=src_path)

def progress_bar(progress, total, color=colorama.Fore.YELLOW):
    percent = 100 * (progress / float(total))
    bar = '█' * int(percent) + "-" * (100 - int(percent))
    print(color + f"\r[ {bar} ] {percent:.2f}%", end="\r")