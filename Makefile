
-include makefile.include

# train model
all: labelme2coco
	python3 recoat_main.py --src $(TRAIN_SRC_DIR) --dst $(TRAIN_DST_DIR)

labelme2coco: labelme2coco.py
	python3 labelme2coco.py $(COCO_SRC_DIR) $(COCO_DST_DIR) --labels ./labels.txt

# generate mask image
mask:
	python3 recoat_json2mask.py --src $(MASK_SRC_DIR) --dst $(MASK_DST_DIR)

# load recoating model and detect images
detect:
	python3 recoat_detect.py --src $(DETECT_SRC_DIR) --dst $(DETECT_DST_DIR) --model $(DETECT_MODEL)

# open recoat detecting system
recoat_system:
	python3 recoat_system.py

# do geometric transform
geometric:
	python3 melt_geometric.py --src $(GEOMETIC_SRC_DIR) --dst $(GEOMETIC_DST_DIR)

# do some logic operation and output the melted defects
defect: geometric
	python3 melt_defect.py --src $(DEFECTS_SRC_DIR) --mask $(DEFECTS_MASK_DIR) --dst $(DEFECTS_DST_DIR)

contour: defect
	python3 melt_contour.py --src $(CONTOUR_SRC_DIR) --dst $(CONTOUR_DST_DIR)

clean:
	-@$(RM) -r $(RESULT_PATH)

# Download labelme2coco.py from wkentaro/labelme repository
labelme2coco.py:
	-@wget -q https://raw.githubusercontent.com/wkentaro/labelme/main/examples/instance_segmentation/labelme2coco.py
	@echo "File labelme2coco.py was patched."

distclean: clean
	-@$(RM) labelme2coco.py
