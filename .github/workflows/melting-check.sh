#!/usr/bin/env bash

# download the used packages
pip3 install colorama opencv-python openpyxl
pip3 install xgboost lightgbm pandas
pip install -U scikit-learn

set -x

# isolate workpieces in image
make find-contour
# glcm
make gen-glcm
# train regression model
make train-model
