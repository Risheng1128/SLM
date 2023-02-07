import os
import argparse

from recoat_trainer import Detector

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help = "source path")
    parser.add_argument("--dst", help = "destination path")
    args = parser.parse_args()

    # create output file
    if not os.path.isdir(args.dst):
        os.makedirs(args.dst)

    trainer = Detector(args.src, args.dst)
    trainer.train()

    print("Train model finish!")