import argparse
import os
import glob
import pydicom

if __name__ == '__main__':
    # generate a unique uid
    default_uid = pydicom.uid.generate_uid()

    parser = argparse.ArgumentParser()
    parser.add_argument('--src',
                        default='./data/ct-example/',
                        help='source image path')
    parser.add_argument('--dst',
                        default='./result/dicom/',
                        help='destination path')
    args = parser.parse_args()

    if not args.src:
        raise AssertionError('Error: source path not found!')

    if not os.path.isdir(args.dst):
        os.makedirs(args.dst)

    image_list = glob.glob(args.src + '*.jpg')
    image_list.sort()

    index = 0
    for i in image_list:
        output_file = args.dst + str(index) + '.dcm'
        # convert jpg to dicom
        os.system("img2dcm " + i + ' ' + output_file)
        # decompress dicom file
        os.system('dicom-decompress --transcode ' + output_file + ' ' + output_file)

        # make the SeriesInstanceUID same
        ds = pydicom.dcmread(output_file)
        ds.SeriesInstanceUID = default_uid
        ds.save_as(output_file)
        index += 1