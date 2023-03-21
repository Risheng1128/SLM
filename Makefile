
-include makefile.include

PYTHON = python3

help:
	@echo "-------------------------- command manual -----------------------------------"
	@echo "|         command         |                   description                   |"
	@echo "|-------------------------+-------------------------------------------------|"
	@echo "| help                    | show command manual                             |"
	@echo "| melt_train_model        | train Mask R-CNN model                          |"
	@echo "| labelme2coco            | convert image to coco dataset                   |"
	@echo "| gen-mask                | generate mask image                             |"
	@echo "| detect-defects          | load recoating model and detect images          |"
	@echo "| recoat-system           | open recoat defects detecting system            |"
	@echo "| geometric-transform     | do geometric transform                          |"
	@echo "| find-contours           | isolate every workpieces in image               |"
	@echo "| gen-glcm                | generate glcm feature from the workpiece images |"
	@echo "| computed-tomography     | view computed tomography via dicom viewer       |"
	@echo "| train-model             | train model to predict material property        |"
	@echo "-----------------------------------------------------------------------------"

# train model
train-mask-rcnn: labelme2coco
	$(PYTHON) recoat_maskrcnn.py --src $(RCNN_SRC_PATH) --dst $(RCNN_DST_PATH)

labelme2coco: labelme2coco.py
	$(PYTHON) labelme2coco.py $(COCO_SRC_PATH) $(COCO_DST_PATH) --labels $(LABEL_PATH)

# generate mask image
gen-mask:
	$(PYTHON) recoat_json2mask.py --src $(MASK_SRC_PATH) --dst $(MASK_DST_PATH)

# load recoating model and detect images
detect-defects:
	$(PYTHON) recoat_detect.py --src $(DETECT_SRC_PATH) --dst $(DETECT_DST_PATH) --model $(DETECT_MODEL)

# open recoat defects detecting system
recoat-system:
	$(PYTHON) recoat_system.py

# do geometric transform
geometric-transform:
	$(PYTHON) melt_geometric.py --src $(GEOMETRIC_SRC_PATH) --dst $(GEOMETRIC_DST_PATH)

# isolate every workpieces in image
find-contours: geometric-transform
	$(PYTHON) melt_contour.py --src $(CONTOUR_SRC_PATH) --mask $(CONTOUR_MASK_PATH) --dst $(CONTOUR_DST_PATH)

# generate glcm feature from the workpiece images
gen-glcm:
	$(PYTHON) melt_glcm.py --src $(GLCM_SRC_PATH) --xlsx $(GLCM_XLSX)

# view computed tomography via dicom view
computed-tomography:
	$(PYTHON) melt_jpg2dicom.py --src $(DICOM_SRC_PATH) --dst $(DICOM_DST_PATH)
	$(PYTHON) melt_dicom_viewer.py

# train xgboost, lightgbm, linear regression and SVR model
train-model:
	$(PYTHON) melt_train_model.py --dst $(BOOST_DST_PATH)

clean:
	-@$(RM) -r $(RESULT)

# Download labelme2coco.py from wkentaro/labelme repository
labelme2coco.py:
	-@wget -q https://raw.githubusercontent.com/wkentaro/labelme/main/examples/instance_segmentation/labelme2coco.py
	@echo "File labelme2coco.py was patched."

distclean: clean
	-@$(RM) labelme2coco.py
