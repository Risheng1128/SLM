import os
import glob
import argparse
import colorama

from recoat_trainer import Detector
from utils import progress_bar

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help = "source path")
    parser.add_argument("--dst", help = "destination path")
    parser.add_argument("--model", help = "model path")
    args = parser.parse_args()

    # create output file
    if not os.path.isdir(args.dst):
        os.makedirs(args.dst)

    # load the parameters into the model
    model = Detector(args.src, args.dst) # we do not specify pretrained=True, i.e. do not load default weights

    # test data
    image_path = args.src + "*.png"
    total = int(len(glob.glob(image_path)))

    # Generating Binary mask and predictions
    for i, image in enumerate(glob.glob(image_path)):
        progress_bar(i, total - 1, color=colorama.Fore.YELLOW)
        model.Save_Mask(image, args.dst, args.model)
        model.Save_Prediction(image, args.dst, args.model)
    print("\nDetect finished !")
    
    # TODO: compute the dice and mAP
    # ground_truth = "./Data/Mask/Test/"
    # prediction = "./output/Test/mask/*.jpg"

    # Dice codfficient calculation
    # dice = dice_folder(ground_truth, prediction)
    # mAP_folder(ground_truth, prediction)
    # print(dice)