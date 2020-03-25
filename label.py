import logging
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os 
from roipoly import RoiPoly

logging.basicConfig(format='%(levelname)s ''%(processName)-10s : %(asctime)s '
                           '%(module)s.%(funcName)s:%(lineno)s %(message)s',
                    level=logging.INFO)
img_dir = "./trainset"
image_set = os.listdir("./trainset/")
#fg_mask stores mask for red regions
#bg_mask stores mask for non-red regions
if not os.path.exists('./fg_mask'):
    os.makedirs('./fg_mask')
if not os.path.exists('./bg_mask'):
    os.makedirs('./bg_mask')

# Create image
for i in range(len(image_set)):
	img_name = image_set[i]
	img = cv2.imread(os.path.join("./trainset",image_set[i]))
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# Show the image
	fig = plt.figure()
	plt.imshow(img, interpolation='nearest', cmap="Greys")
	plt.colorbar()
	plt.title("left click: line segment   right click or double click: close region")
	plt.show(block=False)

	# Let user draw first ROI
	roi1 = RoiPoly(color='r', fig=fig)

	# Show the image with the first ROI
	fig = plt.figure()
	plt.imshow(img, interpolation='nearest', cmap="Greys")
	plt.colorbar()
	roi1.display_roi()
	plt.title('draw second ROI')
	plt.show(block=False)

	# Let user draw second ROI
	roi2 = RoiPoly(color='r', fig=fig)

	# Show the image with both ROIs and their mean values
	# plt.imshow(img, interpolation='nearest', cmap="Greys")
	# plt.colorbar()
	# for roi in [roi1, roi2]:
	#     roi.display_roi()
	#     roi.display_mean(img)
	# plt.title('The two ROIs')
	# plt.show()
	fig = plt.figure()
	plt.imshow(img, interpolation='nearest', cmap="Greys")
	plt.colorbar()
	roi1.display_roi()
	roi2.display_roi()
	plt.title('draw third ROI')
	plt.show(block=False)
	roi3 = RoiPoly(color='b', fig=fig)

	# Show the image with both ROIs and their mean values
	# plt.imshow(img, interpolation='nearest', cmap="Greys")
	# plt.colorbar()
	# for roi in [roi1, roi2, roi3]:
	#     roi.display_roi()
	#     roi.display_mean(gray)
	# plt.title('The two ROIs')
	# plt.show()
	# Show ROI masks
	roi_red = roi1.get_mask(gray) + roi2.get_mask(gray)
	# plt.imshow(roi_red,
	#            interpolation='nearest', cmap="Greys")
	# plt.title('ROI masks of the two ROIs')
	# plt.show()
	roi_background = roi3.get_mask(gray)
	if(roi_red.any()):
		np.save("./fg_mask/"+img_name,roi_red)
	if(roi_background.any()):
		np.save("./bg_mask/"+img_name,roi_background)
	# print(type(roi1.get_mask(img)))
