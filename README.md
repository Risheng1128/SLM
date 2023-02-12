# SLM
A research focusing on [selective laser melting (SLM)](https://en.wikipedia.org/wiki/Selective_laser_melting).

The project is divided into two parts — recoating and melting.

In recoating part, the project applies the [Mask R-CNN](https://github.com/matterport/Mask_RCNN) model with [detectron2](https://ai.facebook.com/tools/detectron2/) to detect the recoating defects. On the other hand, there are three types of recoating defects — powder uneven, powder uncover and scratch. It can be found in [labels.txt](/data/recoat/labels.txt).

In melting part, the research goal is using images to predict the material properties of workpieces. The project focus on the [permeability](https://en.wikipedia.org/wiki/Permeability_(electromagnetism)), [core loss](https://en.wikipedia.org/wiki/Magnetic_core#Core_loss) and [ultimate tensile strength](https://en.wikipedia.org/wiki/Ultimate_tensile_strength).

## Build and Verify
### Recoating
**Install [detectron2](https://ai.facebook.com/tools/detectron2/) and other packages**:
```
pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install detectron2 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
pip3 install labelme opencv-python colorama matplotlib
```

**Intall [labelme](https://github.com/wkentaro/labelme)**: Use labelme to annotate images and `labelme2coco.py` in labelme convert images to coco format.
```
pip install labelme
```

**Train Mask R-CNN model**: Use [recoat_mskrcnn.py](recoat_mskrcnn.py) to train Mask R-CNN model.
```
make
```

**Convert image to [coco dataset](https://cocodataset.org)**: Use labelme2coco.py in [labelme](https://github.com/wkentaro/labelme) project to convert label file to COCO dataset.
```
make labelme2coco
```

**Convert [ground truth](https://en.wikipedia.org/wiki/Ground_truth) mask**: Use [recoat_json2mask.py](recoat_json2mask.py) to generate mask file. The `.json` file will be generated to mask, and we will use the `label.png` to compute dice coefficient.
```
make mask
```

**Load model and detect image**: Use [recoat_detect.py](recoat_detect.py) to load recoating model and detect the images. By default, it uses the `recoat.pth` in folder `model`.
```
make detect
```

**Recoat detecting system**: Use [recoat_system.py](recoat_system.py) to open the recoat detecting system. Before open the system, installing the pyqt5:
```
sudo apt-get install qt5-default
sudo apt-get install qttools5-dev-tools
sudo pip3 install pyqt5
make recoat_system
```

### Melting
**Geometric Transform**: The following image is orignal SLM image and the project annotates it to get the four point in order to make use of geometric transform. The point can be found in [origin.json](data/geometric/origin.json)

![](data/geometric/origin.jpg)

Use [melt_geometric.py](melt_geometric.py) to generate orthographic projection image.
```
make geometric
```

**Isolate workpieces in image**: To have a better analysis for every workpieces, need to isolate them from image. Therefore, using [melt_contour.py](melt_contour.py) to isolate workpiece from image by using mask image (*.bmp).
```
make contour
```

**Gray Level Co-occurrence Matrix (GLCM)**: Because the defects in image is difficult to identify, introducing the GLCM to compute the different features in image.
```
make glcm
```

**Computed Tomography (CT)**: Use computed tomography image make us observe workpiece quality more clearly. In this project, using file `melt_jpg2dicom.py` to convert the `.jpg` files to `.dcm` files which usually are applied in biomedical field. On the other hand, display the dicom image by file `melt_dicom_viewer.py` the reference to [QtVTKDICOMViewer](https://github.com/RasmusRPaulsen/QtVTKDICOMViewer).
```
make computed_tomography
```

## Reference
* Here are tutorial about Detectron2, if you have some question, please watching it
  * [Using Machine Learning with Detectron2](https://www.youtube.com/watch?v=eUSgtfK4ivk&ab_channel=MetaOpenSource)
  * [Detectron2 Custom Object Detection, Custom Instance Segmentation: Part I](https://www.youtube.com/watch?v=ffTURA0JM1Q&ab_channel=TheCodingBug)
  * [Detectron2 Custom Object Detection, Custom Instance Segmentation: Part II](https://www.youtube.com/watch?v=GoItxr16ae8&ab_channel=TheCodingBug)
