import os
import argparse

def convert_to_mask(src_path = "./Data/Example/", dst_path = "./Data/Mask/"):
   if not os.path.exists(dst_path):
      os.makedirs(dst_path)

   dirs = os.listdir(src_path)

   for item in dirs:
      if item.endswith(".json"):
         if os.path.isfile(src_path + item):
            print("C: " + str(item))
            dst = dst_path + str(item).split('.')[0]
            os.system("labelme_json_to_dataset " + src_path + item + " -o " + dst)

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--src", help = "source file path")
   parser.add_argument("--dst", help = "destination of produce path")
   args = parser.parse_args()
   if args.src and args.dst:
      convert_to_mask(args.src, args.dst)
   else:
      convert_to_mask()
   