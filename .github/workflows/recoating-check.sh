#!/usr/bin/env bash

# download the used packages
pip3 install torch==1.10.1+cu111 torchvision==0.12.0+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install detectron2 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
pip3 install opencv-python colorama imgviz labelme pycocotools

set -x

# train Mask R-CNN model
make
# convert ground truth mask
make mask