
-include makefile.include

PYTHON = python3

# train model
all: labelme2coco
	$(PYTHON) recoat_main.py --src $(TRAIN_SRC_DIR) --dst $(TRAIN_DST_DIR)

labelme2coco: labelme2coco.py
	$(PYTHON) labelme2coco.py $(COCO_SRC_DIR) $(COCO_DST_DIR) --labels ./labels.txt

# generate mask image
mask:
	$(PYTHON) recoat_json2mask.py --src $(MASK_SRC_DIR) --dst $(MASK_DST_DIR)

# load recoating model and detect images
detect:
	$(PYTHON) recoat_detect.py --src $(DETECT_SRC_DIR) --dst $(DETECT_DST_DIR) --model $(DETECT_MODEL)

# open recoat detecting system
recoat_system:
	$(PYTHON) recoat_system.py

# do geometric transform
geometric:
	$(PYTHON) melt_geometric.py --src $(GEOMETIC_SRC_DIR) --dst $(GEOMETIC_DST_DIR)

# do some logic operation and output the melted defects
defect: geometric
	$(PYTHON) melt_defect.py --src $(DEFECTS_SRC_DIR) --mask $(DEFECTS_MASK_DIR) --dst $(DEFECTS_DST_DIR)

contour: defect
	$(PYTHON) melt_contour.py --src $(CONTOUR_SRC_DIR) --dst $(CONTOUR_DST_DIR)

computed_tomography:
	$(PYTHON) melt_jpg2dicom.py --src $(DICOM_SRC_DIR) --dst $(DICOM_DST_DIR)
	$(PYTHON) melt_dicom_viewer.py

clean:
	-@$(RM) -r $(RESULT_PATH)

# Download labelme2coco.py from wkentaro/labelme repository
labelme2coco.py:
	-@wget -q https://raw.githubusercontent.com/wkentaro/labelme/main/examples/instance_segmentation/labelme2coco.py
	@echo "File labelme2coco.py was patched."

distclean: clean
	-@$(RM) labelme2coco.py
