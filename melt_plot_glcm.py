import argparse
import cv2 as cv
import numpy as np
import os

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
	h,w = img.shape
	# digitize
	bins = np.linspace(vmin, vmax+1, levels+1)
	gl1 = np.digitize(img, bins) - 1

	# make shifted image
	dx = distance*np.cos(np.deg2rad(angle))
	dy = distance*np.sin(np.deg2rad(-angle))
	mat = np.array([[1.0,0.0,-dx], [0.0,1.0,-dy]], dtype=np.float32)
	gl2 = cv.warpAffine(gl1, mat, (w,h), flags=cv.INTER_NEAREST, borderMode=cv.BORDER_REPLICATE)

	# make glcm
	glcm = np.zeros((levels, levels), dtype=np.uint8)
	for i in range(levels):
		for j in range(levels):
			mask = ((gl1==i) & (gl2==j))
			glcm[i,j] = mask.sum()

	return glcm / glcm.sum()

def compute_energy(glcm):
	energy = 0
	row, col = glcm.shape
	for i in range(row):
		for j in range(col):
			energy += glcm[i, j] ** 2
	return energy

def compute_entropy(glcm):
	entropy = 0
	row, col = glcm.shape
	for i in range(row):
		for j in range(col):
			if glcm[i, j]:
				entropy += -np.log(glcm[i, j]) * glcm[i, j]
	return entropy

def compute_contrast(glcm):
	contrast = 0
	row, col = glcm.shape
	for i in range(row):
		for j in range(col):
			contrast += glcm[j, j] * ((i - j) ** 2)
	return contrast

# compute inverse differential moment (IDM)
def compute_idm(glcm):
	idm = 0
	row, col = glcm.shape
	for i in range(row):
		for j in range(col):
			idm += glcm[i, j] / (1 + (i - j) ** 2)
	return idm

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--src', help = 'source image path')
	parser.add_argument('--dst', help = 'destination of folder path')
	args = parser.parse_args()

	if not args.src:
	    raise AssertionError('Source path not found!!')

	if not os.path.isdir(args.dst):
		print('No directory. Construct one')
		os.makedirs(args.dst)

	img = cv.imread(args.src)
	img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	cv.imwrite(args.dst + 'origin.jpg', img)

	total_energy = 0
	total_entropy = 0
	total_contrast = 0
	total_idm = 0
	for i in [0, 45, 90, 135]:
		glcm = get_glcm(img, angle=i)
		energy = compute_energy(glcm)
		entropy = compute_entropy(glcm)
		contrast = compute_contrast(glcm)
		idm = compute_idm(glcm)

		total_energy += energy
		total_entropy += entropy
		total_contrast += contrast
		total_idm += idm

		print('----------------------------------')
		print("energy = ", energy)
		print("entropy = ", entropy)
		print("contrast = ", contrast)
		print("idm = ", idm)

	print('----------------------------------')
	print("avr energy = ", total_energy / 4)
	print("avr entropy = ", total_entropy / 4)
	print("avr contrast = ", total_contrast / 4)
	print("avr idm = ", total_idm / 4)