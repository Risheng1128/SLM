import os
import argparse

def convert_to_mask(src_path='./data/recoat/', dst_path='./result/mask/'):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    dirs = os.listdir(src_path)

    for item in dirs:
        if item.endswith(".json"):
            if os.path.isfile(src_path + item):
                print("C: " + str(item))
                dst = dst_path + str(item).split('.')[0]
                os.system("labelme_json_to_dataset " +
                          src_path + item + " -o " + dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',
                        default='./data/recoat/',
                        help='source image path')
    parser.add_argument('--dst',
                        default='./result/mask/',
                        help='destination path')
    args = parser.parse_args()

    convert_to_mask(args.src, args.dst)
