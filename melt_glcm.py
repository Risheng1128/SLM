import argparse
import cv2 as cv
import numpy as np
import os
import openpyxl
import glob

# compute gray-level co-occurrence matrix
def get_glcm(img, vmin=0, vmax=255, levels=8, distance=1.0, angle=0.0):
    '''
    Parameters
    ----------
    img: array_like, shape=(h,w), dtype=np.uint8
        input image
    vmin: int
        minimum value of input image
    vmax: int
        maximum value of input image
    levels: int
        number of grey-levels of GLCM
    distance: float
        pixel pair distance offsets [pixel] (1.0, 2.0, and etc.)
    angle: float
        pixel pair angles [degree] (0.0, 30.0, 45.0, 90.0, and etc.)

    Returns
    -------
    Grey-level co-occurrence matrix
    shape = (levels, levels)
    '''
    h, w = img.shape
    # digitize
    bins = np.linspace(vmin, vmax + 1, levels + 1)
    gl1 = np.digitize(img, bins) - 1

    # make shifted image
    dx = distance * np.cos(np.deg2rad(angle))
    dy = distance * np.sin(np.deg2rad(-angle))
    mat = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
    gl2 = cv.warpAffine(gl1, mat, (w, h), flags=cv.INTER_NEAREST,
                        borderMode=cv.BORDER_REPLICATE)

    # make glcm
    glcm = np.zeros((levels, levels), dtype=np.uint8)

    for i in range(levels):
        for j in range(levels):
            mask = ((gl1 == i) & (gl2 == j))
            glcm[i, j] = mask.sum()

    # eliminate the affect of background
    glcm[:, 0] = 0
    glcm[0, :] = 0

    # normalize
    return glcm / glcm.sum()

# compute mean feature
def compute_mean(glcm):
    mean_x = 0
    mean_y = 0
    row, col = glcm.shape

    for i in range(row):
        for j in range(col):
            mean_x += glcm[i, j] * i
            mean_y += glcm[i, j] * j
    return mean_x, mean_y

# compute variance feature
def compute_variance(glcm, mean_x, mean_y):
    variance_x = 0
    variance_y = 0
    row, col = glcm.shape

    for i in range(row):
        for j in range(col):
            variance_x += glcm[i, j] * ((i - mean_x) ** 2)
            variance_y += glcm[i, j] * ((i - mean_y) ** 2)
    return variance_x, variance_y

# compute standard deviation feature
def compute_standard_deviation(glcm, variance_x, variance_y):
    return np.sqrt(variance_x), np.sqrt(variance_y)

# compute cluster prominence feature
def compute_cluster_prominence(glcm, mean_x, mean_y):
    cluster_prominence = 0
    row, col = glcm.shape

    for i in range(row):
        for j in range(col):
            cluster_prominence += ((i + j - mean_x - mean_y) ** 4) * glcm[i, j]
    return cluster_prominence

# compute cluster shade feature
def compute_cluster_shade(glcm, mean_x, mean_y):
    cluster_shade = 0
    row, col = glcm.shape

    for i in range(row):
        for j in range(col):
            cluster_shade += ((i + j - mean_x - mean_y) ** 3) * glcm[i, j]
    return cluster_shade

# compute cluster tendency feature
def compute_cluster_tendency(glcm, mean_x, mean_y):
    cluster_tendency = 0
    row, col = glcm.shape

    for i in range(row):
        for j in range(col):
            cluster_tendency += ((i + j - mean_x - mean_y) ** 2) * glcm[i, j]
    return cluster_tendency

# compute autocorrelation feature
def compute_autocorrelation(glcm):
    autocorrelation = 0
    row, col = glcm.shape

    for i in range(row):
        for j in range(col):
            autocorrelation += glcm[i, j] * i * j
    return autocorrelation

# compute correlation feature
def compute_correlation(glcm, mean_x, mean_y, variance_x, variance_y):
    correlation = 0
    row, col = glcm.shape

    for i in range(row):
        for j in range(col):
            correlation += glcm[i, j] * (i - mean_x) * \
                (j - mean_y) / np.sqrt(variance_x * variance_y)
    return correlation

# compute dissimilarity feature
def compute_dissimilarity(glcm):
    dissimilarity = 0
    row, col = glcm.shape

    for i in range(row):
        for j in range(col):
            dissimilarity += glcm[i, j] * np.abs(i - j)
    return dissimilarity

# compute energy feature
def compute_energy(glcm):
    energy = 0
    row, col = glcm.shape

    for i in range(row):
        for j in range(col):
            energy += glcm[i, j] ** 2
    return energy

# compute entropy feature
def compute_entropy(glcm):
    entropy = 0
    row, col = glcm.shape

    for i in range(row):
        for j in range(col):
            if glcm[i, j]:
                entropy += -np.log(glcm[i, j]) * glcm[i, j]
    return entropy

# compute contrast feature
def compute_contrast(glcm):
    contrast = 0
    row, col = glcm.shape

    for i in range(row):
        for j in range(col):
            contrast += glcm[j, j] * ((i - j) ** 2)
    return contrast

# compute inverse differential moment (IDM) feature
def compute_idm(glcm):
    idm = 0
    row, col = glcm.shape

    for i in range(row):
        for j in range(col):
            idm += glcm[i, j] / (1 + (i - j) ** 2)
    return idm

def create_header(sheet):
    sheet.cell(1, 1).value = 'layer'
    sheet.cell(1, 2).value = 'energy'
    sheet.cell(1, 3).value = 'entropy'
    sheet.cell(1, 4).value = 'contrast'
    sheet.cell(1, 5).value = 'idm'
    sheet.cell(1, 6).value = 'autocorrelation'
    sheet.cell(1, 7).value = 'mean_x'
    sheet.cell(1, 8).value = 'mean_y'
    sheet.cell(1, 9).value = 'variance_x'
    sheet.cell(1, 10).value = 'variance_y'
    sheet.cell(1, 11).value = 'standard_deviation_x'
    sheet.cell(1, 12).value = 'standard_deviation_y'
    sheet.cell(1, 13).value = 'correlation'
    sheet.cell(1, 14).value = 'dissimilarity'

def store_data(sheet, feature, layer_num):
    row = layer_num + 2
    sheet.cell(row, 1).value = 'layer' + str(layer_num)
    sheet.cell(row, 2).value = feature["energy"]
    sheet.cell(row, 3).value = feature["entropy"]
    sheet.cell(row, 4).value = feature["contrast"]
    sheet.cell(row, 5).value = feature["idm"]
    sheet.cell(row, 6).value = feature["autocorrelation"]
    sheet.cell(row, 7).value = feature["mean_x"]
    sheet.cell(row, 8).value = feature["mean_y"]
    sheet.cell(row, 9).value = feature["variance_x"]
    sheet.cell(row, 10).value = feature["variance_y"]
    sheet.cell(row, 11).value = feature["standard_deviation_x"]
    sheet.cell(row, 12).value = feature["standard_deviation_y"]
    sheet.cell(row, 13).value = feature["correlation"]
    sheet.cell(row, 14).value = feature["dissimilarity"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',
                        default='./data/ct-example/',
                        help='source image path')
    parser.add_argument('--xlsx',
                        default='glcm.xlsx',
                        help='xlsx filename')
    args = parser.parse_args()

    if not args.src:
        raise AssertionError('Source path not found!!')

    item_list = os.listdir(args.src)
    item_list = sorted([i for i in item_list])
    item_list = [args.src + i + '/' for i in item_list]
    item_num = len(item_list)

    wb = openpyxl.Workbook()
    for i in range(item_num):
        layer_list = list(
            enumerate(glob.glob(os.path.join(item_list[i], "*.jpg"))))
        layer_list = sorted([i[1] for i in layer_list])
        layer_num = len(layer_list)

        sheet = wb.create_sheet('item' + str(i))
        create_header(sheet)

        for j in range(layer_num):
            img = cv.imread(layer_list[j])
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            feature = {"autocorrelation": 0,
                       "mean_x": 0,
                       "mean_y": 0,
                       "variance_x": 0,
                       "variance_y": 0,
                       "standard_deviation_x": 0,
                       "standard_deviation_y": 0,
                       "cluster_prominence": 0,
                       "cluster_shade": 0,
                       "cluster_tendency": 0,
                       "contrast": 0,
                       "correlation": 0,
                       "dissimilarity": 0,
                       "energy": 0,
                       "entropy": 0,
                       "idm": 0}

            for z in [0, 45, 90, 135]:
                glcm = get_glcm(img, distance=1, angle=z, levels=8)

                mean_x, mean_y = compute_mean(glcm)
                feature["mean_x"] += mean_x
                feature["mean_y"] += mean_y

                variance_x, variance_y = compute_variance(glcm, mean_x, mean_y)
                feature["variance_x"] += variance_x
                feature["variance_y"] += variance_y

                standard_deviation_x, standard_deviation_y = \
                    compute_standard_deviation(glcm, variance_x, variance_y)
                feature["standard_deviation_x"] += standard_deviation_x
                feature["standard_deviation_y"] += standard_deviation_y

                feature["contrast"] += compute_contrast(glcm)
                feature["autocorrelation"] += compute_autocorrelation(glcm)
                feature["correlation"] += compute_correlation(
                    glcm, mean_x, mean_y, variance_x, variance_y)
                feature["dissimilarity"] += compute_dissimilarity(glcm)
                feature["energy"] += compute_energy(glcm)
                feature["entropy"] += compute_entropy(glcm)
                feature["idm"] += compute_idm(glcm)

            for key, value in feature.items():
                feature[key] = value / 4

            store_data(sheet, feature, j)
            wb.save(args.xlsx)
            print('finish layer: ', j + 1)

        print('finish item: ', i + 1)
