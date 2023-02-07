# Detect_SLM_Defects

## Detectron2 Installation
Ubuntu 20.04 LTS:
```
pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install detectron2 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
pip3 install labelme opencv-python colorama matplotlib
```

## Download Detect_SLM_Defects
```
git clone https://github.com/Risheng1128/Detect_SLM_Defects.git
cd Detect_SLM_Defects
```

## How to use
The project is divided into two parts —  recoating and melting.

In recoating part, the project uses the Mask-R-CNN model with detectron2 to detect the recoating defects. On the other hand, there are three types of recoating defects — powder uneven, powder_uncover and scratch. It can be found in [labels.txt](labels.txt)

In melting part, the project is still in experimental stage.

### Labelme
Using Labelme to annotate images and `labelme2coco.py` in labelme convert images to COCO Format.

Download `labelme2coco.py`:
```
wget -q https://raw.githubusercontent.com/wkentaro/labelme/main/examples/instance_segmentation/labelme2coco.py
```

### Recoating Part
The following is the different commands in recoating part.

#### Train recoating model
Use [recoat_mskrcnn.py](recoat_mskrcnn.py) to train Mask R-CNN model

Format:
```
python3 recoat_mskrcnn.py --src <source image path> --dst <destination image path>
```

Also can use the following command
```
make
```

#### Convert Image to COCO Dataset
Using labelme2coco.py in labelme project to convert label file to COCO dataset

Format:
```
python3 labelme2coco.py <input image path> <output image path> --labels <label text>
```

labels.txt Format:
```
__ignore__
class1
class2
...
```

Also can use the following command
```
make labelme2coco
```

#### Convert ground truth mask
Using [recoat_json2mask.py](recoat_json2mask.py) to generate mask file

The `.json` file will be generated to mask, and we will use the `label.png` to compute dice coefficient

Format:
```
python3 recoat_json2mask.py --src <source image path> --dst <destination image path>
```

Example:
```
python3 recoat_json2mask.py --src ./Data/Recoat/ --dst ./Result/Mask/
```

Also can use the following command
```
make mask
```

#### Load model and detect image
Using [recoat_detect.py](recoat_detect.py) to load recoating model and detect the images. By default, it uses the `recoat.pth` model in folder `Model`

Format:
```
python3 recoat_detect.py --src <source image path> --dst <destination image path> --model <model path>
```

Example:
```
python3 recoat_detect.py --src ./Data/Recoat/ --dst ./Result/Detect/ --model ./Model/recoat.pth
```

Also can use the following command
```
make detect
```

#### Recoat Detecting System
Using [recoat_system.py](recoat_system.py) to open the recoat detecting system

First, download the pyqt5
```
$ sudo apt-get install qt5-default
$ sudo apt-get install qttools5-dev-tools
$ sudo pip3 install pyqt5
```

And use the following command.

Example:
```
python3 recoat_system.py
```

Also can use the following command
```
make recoat_system
```

### Melting Part
The following is the different commands in melting part.

#### Geometric Transform
The following image is orignal SLM image and the project annotates it to get the four point in order to make use of geometric transform. The point can be found in [origin.json](Data/Geometric/origin.json)

![](Data/Geometric/origin.jpg)

Using [melt_geometric.py](melt_geometric.py) to generate orthographic projection image.

Format:
```
python3 melt_geometric.py --src <source image path> --dst <destination image path>
```

Example:
```
python3 melt_geometric.py --src ./Data/Melt/ --dst ./Result/Melt_Geometric/
```

Also can use the following command
```
make geometric
```

#### Isolate Every Workpiece in Image
To have a better analysis for every workpieces, need to isolate them from image. Therefore, using [melt_contour.py](melt_contour.py) to isolate workpiece from image by using mask image (*.bmp).

Format:
```
python3 melt_contour.py --src <source image path> --mask <mask image path> --dst <destination image path>
```

Also can use the following command
```
make contour
```

#### Gray Level Co-occurrence Matrix (GLCM)
Because the defects in image is difficult to identify, introducing the GLCM to compute the different features in image.

Format:
```
python3 melt_plot_glcm.py --src <source image path> --xlsx <xlsx filename>
```

Also can use the following command
```
make glcm
```

#### Computed Tomography (CT)
Using computed tomography image make us observe workpiece quality more clearly. In this project, using file `melt_jpg2dicom.py` to convert the `.jpg` files to `.dcm` files which usually are applied in biomedical field. On the other hand, display the dicom image by file `melt_dicom_viewer.py` the reference to [QtVTKDICOMViewer](https://github.com/RasmusRPaulsen/QtVTKDICOMViewer).

Format:
```
python3 melt_jpg2dicom.py --src <source image path> --dst <destination image path>
python3 melt_dicom_viewer.py
```

Also can use the following command
```
make computed_tomography
```

## Reference
* if you do not want to use Mask R-CNN, please go to [model_zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md)

* Here are tutorial, if you have some question, please watching it
  * [Using Machine Learning with Detectron2](https://www.youtube.com/watch?v=eUSgtfK4ivk&ab_channel=MetaOpenSource)
  * [DETECTRON2 Custom Object Detection, Custom Instance Segmentation: Part I](https://www.youtube.com/watch?v=ffTURA0JM1Q&ab_channel=TheCodingBug)
  * [DETECTRON2 Custom Object Detection, Custom Instance Segmentation: Part II](https://www.youtube.com/watch?v=GoItxr16ae8&ab_channel=TheCodingBug)
* if you want to use [Swin Transformer](https://arxiv.org/pdf/2111.09883.pdf), please custom backbone and config file, [here](https://github.com/xiaohu2015/SwinT_detectron2) is an example
