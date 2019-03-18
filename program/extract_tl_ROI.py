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

class RotateTrafficLight:
	def __init__(self):
		subprocess.call(
			"rm -rf ./extracted_tl_image ; mkdir -p ./extracted_tl_image ; mkdir -p ./extracted_tl_image/train ; mkdir -p ./extracted_tl_image/val", shell=True)
		self.train_ratio = 0.8
	
	def Rotate(self, im_dir, test):
		im_list = glob.glob(im_dir+"/*.jpg")
		im_count = len(im_list)

		im_buf = []
		im_lst_buf = []

		print("Image Num = "+str(im_count))

		image_index = 0
		train_list_str = ""
		val_list_str = ""

		for im_name in im_list:
			label_name = os.path.splitext(im_name)[0]+".txt"
			_label_name = "./extracted_tl_image/"+os.path.basename(label_name)
			if os.path.isfile(label_name):
				print(im_name)

				img = cv2.imread(im_name)
				HEIGHT, WIDTH = img.shape[:2]

				save_flag = False
				labels = ""

				lines = []
				with open(label_name, "r") as f:
					lines = f.readlines()

				for line in lines:
					_line = line.split()
					signal = _line[0]
					x = float(_line[1])*WIDTH
					y = float(_line[2])*HEIGHT
					width = float(_line[3])*WIDTH
					height = float(_line[4])*HEIGHT

					im_tr = img[int(y-height/2):int(y+height/2), int(x-width/2):int(x+width/2)]
					if im_tr.shape[0] > 0 and im_tr.shape[1] > 0:
						#imname = "/extracted_tl_image/im"+str(image_index)+".jpg"
						_im_tr = cv2.resize(im_tr, dsize=(224,224))
						#_im_tr = im_tr
						im_buf.append(_im_tr)
						im_lst_buf.append(str(image_index)+"\t"+signal+"\t")
						#cv2.imwrite("."+imname, im_tr)
						#list_str += str(image_index)+"\t"+signal+"\t"+os.getcwd()+imname+"\n"
						image_index +=1

		p = list(zip(im_buf, im_lst_buf))
		random.shuffle(p)
		im_buf_t, im_lst_buf_t = zip(*p)
		im_buf = list(im_buf_t)
		im_lst_buf = list(im_lst_buf_t)


		if test == 0:
			train_num = self.train_ratio*len(im_buf)

			train_count = 0
			test_count = 0
			for i,_buf in enumerate(im_buf):
				if train_count <= train_num:
					imname = "/extracted_tl_image/train/im"+str(train_count)+".jpg"
					cv2.imwrite("."+imname, _buf)
					train_list_str += im_lst_buf[i]+os.getcwd()+imname+"\n"
					train_count+=1
				else:
					imname = "/extracted_tl_image/val/im"+str(test_count)+".jpg"
					cv2.imwrite("."+imname, _buf)
					val_list_str += im_lst_buf[i]+os.getcwd()+imname+"\n"
					test_count+=1

			with open("./extracted_tl_image/train/image.lst", "w") as fp:
				fp.write(train_list_str)
			with open("./extracted_tl_image/val/image.lst", "w") as fp:
				fp.write(val_list_str)

			subprocess.call("python /usr/local/lib/python2.7/dist-packages/mxnet/tools/im2rec.py ./extracted_tl_image/train/image.lst ./extracted_tl_image/train/ --train-ratio=1.0", shell=True)
			subprocess.call("python /usr/local/lib/python2.7/dist-packages/mxnet/tools/im2rec.py ./extracted_tl_image/val/image.lst ./extracted_tl_image/val/ --train-ratio=1.0", shell=True)

		else:
			test_count = 0
			test_list_str = ""
			subprocess.call("rm -rf test_tl ; mkdir ./test_tl", shell=True)
			for i,_buf in enumerate(im_buf):
				
				imname = "./test_tl/im"+str(test_count)+".jpg"
				test_count+=1
				cv2.imwrite(imname, _buf)
				test_list_str += im_lst_buf[i]+os.getcwd()+imname+"\n"
			with open("./test_tl/image.lst", "w") as fp:
				fp.write(test_list_str)

if __name__=='__main__':
	args = []
	args = sys.argv
	im_dir = args[1]
	test = args[2]
	rtfl = RotateTrafficLight()
	rtfl.Rotate(im_dir, test)
