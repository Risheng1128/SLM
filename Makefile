-include makefile.include

PYTHON = python3

help:
	@echo "+------------------------- command manual ----------------------------------+"
	@echo "|         command         |                   description                   |"
	@echo "|-------------------------+-------------------------------------------------|"
	@echo "| help                    | show command manual                             |"
	@echo "| geometric-transform     | do geometric transform                          |"
	@echo "| find-contours           | isolate every workpieces in image               |"
	@echo "| glcm                    | generate glcm feature from the workpiece images |"
	@echo "| computed-tomography     | view computed tomography via dicom viewer       |"
	@echo "| train-model             | train model to predict material property        |"
	@echo "+---------------------------------------------------------------------------+"

# do geometric transform
geometric-transform:
	$(PYTHON) geometric.py --src $(GEOMETRIC_SRC_PATH) --dst $(GEOMETRIC_DST_PATH)

# isolate every workpieces in image
find-contours: geometric-transform
	$(PYTHON) contour.py --src $(CONTOUR_SRC_PATH) --mask $(CONTOUR_MASK_PATH) --dst $(CONTOUR_DST_PATH)

# generate glcm feature from the workpiece images
glcm:
	$(PYTHON) glcm.py --src $(GLCM_SRC_PATH) --xlsx $(GLCM_XLSX)

# view computed tomography via dicom view
computed-tomography:
	$(PYTHON) jpg2dicom.py --src $(DICOM_SRC_PATH) --dst $(DICOM_DST_PATH)
	$(PYTHON) dicom_viewer.py

# train xgboost, lightgbm, linear regression and SVR model
train-model:
	$(PYTHON) train_model.py --dst $(BOOST_DST_PATH)

clean:
	-@$(RM) -r $(RESULT)
