#!/usr/bin/env bash

# download the used packages
pip3 install colorama opencv-python openpyxl

set -x

# isolate workpieces in image
make contour
# glcm
make glcm