import cv2
from skimage import data, util
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import numpy as np
import os
import time


def create_samples(img, mask):
    img_flat = img.reshape(-1,3)
    mask = mask.reshape(-1)
    class_pixels = img_flat[np.where(mask>0)[0]]

    return class_pixels

trainset = os.listdir("./trainset/")

data_1 = []
data_0 = []
for i in range(len(trainset)):
    st = time.time()
    img = cv2.imread(os.path.join("./trainset",trainset[i]))
    try: 
        mask_red = np.load("./fg_mask/"+trainset[i]+".npy")
        data_1.append(create_samples(img,mask_red))
    except:
        pass
    try:
        mask_bg = np.load("./bg_mask/"+trainset[i]+".npy")
        
        data_0.append(create_samples(img,mask_bg))
    except:
        pass

print("pixels extracted")

mat_1 = np.vstack(data_1)/255.
mat_0 = np.vstack(data_0)/255.
np.random.shuffle(mat_1)
np.random.shuffle(mat_0)


np.save("class_1_rgb.npy",mat_1)
np.save("class_0_rgb.npy",mat_0)

print("done!!")
