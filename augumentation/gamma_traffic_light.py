#!/usr/bin/env python
import os.path
import sys
import message_filters
import subprocess
import cv2
import numpy as np
import math
import re
import glob
import random

class ContrastTrafficLight:
	def __init__(self):
		subprocess.call(
			"rm -rf ./gamma_image ; mkdir -p ./gamma_image", shell=True)

		self.LUT_DARK = np.arange(256, dtype = 'uint8')
		self.LUT_LIGHT = np.arange(256, dtype = 'uint8')

		gamma_1 = 0.3
		gamma_2 = 3.0

		for i in range(256):
			self.LUT_DARK[i] = 0
			self.LUT_DARK[i] = 255*pow(float(i)/255, 1.0/gamma_1)

		for i in range(256):
			self.LUT_LIGHT[i] = 0
			self.LUT_LIGHT[i] = 255*pow(float(i)/255, 1.0/gamma_2)
		

	
	def Contrast(self, im_dir):
		im_list = glob.glob(im_dir+"/*.jpg")
		im_count = len(im_list)
		print("Image Num = "+str(im_count))
		for im_name in im_list:
			label_name = os.path.splitext(im_name)[0]+".txt"
			_label_name = "./gamma_image/"+os.path.basename(label_name)
			if os.path.isfile(label_name):
				print(im_name)

				img = cv2.imread(im_name)
				high_cont_img = cv2.LUT(img, self.LUT_DARK)
				low_cont_img = cv2.LUT(img, self.LUT_LIGHT)
				high_prefix = "./gamma_image/"+"dark_"+os.path.splitext(os.path.basename(im_name))[0]
				low_prefix = "./gamma_image/"+"light_"+os.path.splitext(os.path.basename(im_name))[0]
				cv2.imwrite(high_prefix+".jpg", high_cont_img)
				cv2.imwrite(low_prefix+".jpg", low_cont_img)
				

				subprocess.call("cp "+label_name+" "+high_prefix+".txt", shell=True)
				subprocess.call("cp "+label_name+" "+low_prefix+".txt", shell=True)

if __name__=='__main__':
	args = []
	args = sys.argv
	im_dir = args[1]
	ctfl = ContrastTrafficLight()
	ctfl.Contrast(im_dir)
