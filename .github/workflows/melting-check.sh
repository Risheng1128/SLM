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

# check grid search and random search
python3 tests/lightgbm_hyper.py
python3 tests/xgboost_hyper.py
python3 tests/svr_hyper.py
python3 tests/logistic_hyper.py
