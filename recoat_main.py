from recoat_trainer import *

import glob
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help = "source folder path")
    parser.add_argument("--dst", help = "destination of folder path")
    args = parser.parse_args()

    # create output file
    if not os.path.isdir(args.dst):
        os.makedirs(args.dst)

    trainer = Detector(args.src, args.dst)
    trainer.train()

    print("Train model finish!")