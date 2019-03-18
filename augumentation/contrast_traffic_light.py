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
			"rm -rf ./contrast_image ; mkdir -p ./contrast_image", shell=True)

		self.min_table = 50
		self.max_table = 205
		self.diff_table = self.max_table - self.min_table

		self.LUT_HC = np.arange(256, dtype = 'uint8')
		self.LUT_LC = np.arange(256, dtype = 'uint8')

		for i in range(0, self.min_table):
			self.LUT_HC[i] = 0
		for i in range(self.min_table, self.max_table):
			self.LUT_HC[i] = 255*(i - self.min_table) / self.diff_table
		for i in range(self.max_table, 255):
			self.LUT_HC[i] = 255

		for i in range(256):
			self.LUT_LC[i] = self.min_table + i * (self.diff_table) / 255
		

	
	def Contrast(self, im_dir):
		im_list = glob.glob(im_dir+"/*.jpg")
		im_count = len(im_list)
		print("Image Num = "+str(im_count))
		for im_name in im_list:
			label_name = os.path.splitext(im_name)[0]+".txt"
			_label_name = "./contrast_image/"+os.path.basename(label_name)
			if os.path.isfile(label_name):
				print(im_name)

				img = cv2.imread(im_name)
				high_cont_img = cv2.LUT(img, self.LUT_HC)
				low_cont_img = cv2.LUT(img, self.LUT_LC)
				high_prefix = "./contrast_image/"+"highcont_"+os.path.splitext(os.path.basename(im_name))[0]
				low_prefix = "./contrast_image/"+"lowcont_"+os.path.splitext(os.path.basename(im_name))[0]
				cv2.imwrite(high_prefix+".jpg", high_cont_img)
				#cv2.imwrite(low_prefix+".jpg", low_cont_img)
				

				subprocess.call("cp "+label_name+" "+high_prefix+".txt", shell=True)
				#subprocess.call("cp "+label_name+" "+low_prefix+".txt", shell=True)

if __name__=='__main__':
	args = []
	args = sys.argv
	im_dir = args[1]
	ctfl = ContrastTrafficLight()
	ctfl.Contrast(im_dir)
