'''
ECE276A WI20 HW1
Stop Sign Detector
'''

import os, cv2
from skimage.measure import label, regionprops
import numpy as np

class StopSignDetector():
	def __init__(self):
		'''
			Initilize your stop sign detector with the attributes you need,
			e.g., parameters of your classifier
		'''

		self.mu1 = np.load("./mu1.npy")
		self.mu0 = np.load("./mu0.npy")
		self.cov1 = np.load("./cov1.npy")
		self.cov0 = np.load("./cov0.npy")
		self.prior1 = np.load("./prior1.npy")
		self.prior0 = np.load("./prior0.npy")

	def calculate_likelihood(self,x,mu,cov,prior):
        #calculates log likelihood of a pixel x for a given gaussian distribution
	    term1 = x - mu
	    t = 0.5*(term1)@np.linalg.inv(cov)
	    term2 = np.multiply(0.5*(term1)@np.linalg.inv(cov),term1).sum(axis = 1)
	    return np.log(prior) - 0.5*np.log(np.linalg.det(cov)) - term2

	def predict(self,x):
        #predicts class of a pixel x
	    y_pred = self.calculate_likelihood(x,self.mu1,self.cov1,self.prior1) > self.calculate_likelihood(x,self.mu0,self.cov0,self.prior0) 
	    return y_pred.reshape(-1)
    
	def create_mask(self,img):
        #Compute the segmentation mask for an image
	    w, h = img.shape[0], img.shape[1]
	    mask = np.zeros((w,h))
	    for i in range(w):
	        mask[i,:] = self.predict(img[i,:])

	    return mask

	def aspectratio(self,cnt):
        #calculate aspect ratio of a countour
	    x,y,w,h = cv2.boundingRect(cnt)
	    return float(w)/h

	def extent(self,cnt):
        #calculate extent for a countour
	    area = cv2.contourArea(cnt)
	    x,y,w,h = cv2.boundingRect(cnt)
	    rect_area = w*h 
	    return float(area)/rect_area

	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
		'''
		# YOUR CODE HERE

		mask = self.create_mask(img)

		return mask

	def get_bounding_box(self, img):
		'''
			Find the bounding box of the stop sign
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
				is from left to right in the image.
				
			Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
		'''
		# YOUR CODE HERE
		
		def get_val(lst):
			return lst[0]
		out_test = self.segment_image(img/255.)
		mask = out_test.astype(np.uint8)
		cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

		eps = (img.shape[0]*img.shape[1])/500
		cont = [x for x in cnts if cv2.contourArea(x) > eps]
		bbox_list = []
		for cnt in cont:
		    hull = cv2.convexHull(cnt)
		    approx = cv2.approxPolyDP(hull,0.0075*cv2.arcLength(cnt,True),True)  
		    aspect_ratio = self.aspectratio(hull)
		    extent_ = self.extent(approx)

		    if  ( aspect_ratio > 0.75 and aspect_ratio <1.5 and extent_ > 0.7 ):
		        if(len(approx) >=7 and len(approx) <=9):
		            rect = cv2.boundingRect(approx)
		            x,y,w,h = rect
		            bbox_list.append([x,img.shape[0]-y-h,x+w,img.shape[0]-y])
                    
		return sorted(bbox_list,key = get_val) 



if __name__ == '__main__':
	folder = "trainset"
	my_detector = StopSignDetector()
	for filename in os.listdir(folder):
		# read one test image
		img = cv2.imread(os.path.join(folder,filename))
		mask_img = my_detector.segment_image(img)
		boxes = my_detector.get_bounding_box(img)
		cv2.namedWindow("output", cv2.WINDOW_NORMAL)                    
		imS = cv2.resize(img, (960, 540))
		maskS = cv2.resize(mask_img, (960, 540))                   
  
		cv2.imshow('image', maskS)
		cv2.waitKey(0)
		cv2.imshow("output", imS)                           
		cv2.waitKey(0)   
		cv2.destroyAllWindows()
		print(boxes)

		#Display results:
		#(1) Segmented images
		#	 mask_img = my_detector.segment_image(img)
		#(2) Stop sign bounding box
		#    boxes = my_detector.get_bounding_box(img)
		#The autograder checks your answers to the functions segment_image() and get_bounding_box()
		#Make sure your code runs as expected on the testset before submitting to Gradescope

