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
			"rm -rf ./rotated_image ; mkdir -p ./rotated_image", shell=True)
	
	def Rotate(self, im_dir):
		im_list = glob.glob(im_dir+"/*.jpg")
		im_count = len(im_list)
		print("Image Num = "+str(im_count))
		for im_name in im_list:
			label_name = os.path.splitext(im_name)[0]+".txt"
			_label_name = "./rotated_image/"+os.path.basename(label_name)
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

					angle = 0

					while True:
						angle = random.randint(-3,3)
						if angle != 0:
							if abs(angle) == 1:
								angle += angle
							break

					angle_rad = math.radians(angle)
					S = math.sin(angle_rad)
					C = math.cos(angle_rad)
					_width = width*C+height*S
					_height=width*S+height*C
					M = cv2.getRotationMatrix2D((int(width/2), int(height/2)), angle, 1.0)
					M[0][2]+=(_width - width)/2
					M[1][2]+=(_height-height)/2

					correct_image = False
					
					if im_tr.shape[0] > 0 and im_tr.shape[1] > 0:
						im_tr = cv2.warpAffine(im_tr, M, (int(_width), int(_height)))
						_lty = y-_height/2
						_rby = _lty + _height
						_ltx = x-_width/2
						_rbx = _ltx + _width

						if _lty>0 and _ltx > 0 and _rby < HEIGHT and _rbx < WIDTH: 
							im_tr = cv2.resize(im_tr, dsize =(int(_rbx)-int(_ltx), int(_rby) - int(_lty)) )
							#print("("+str(int(_lty))+", "+str(int(_rby))+"), ("+str(int(_ltx))+", "+str(int(_rbx))+")")
							img[int(_lty):int(_rby), int(_ltx):int(_rbx)] = im_tr
							labels += signal+" "+_line[1]+" "+_line[2]+" "+str(_width/WIDTH)+" "+str(_height/HEIGHT)+"\n"
							correct_image = True
							save_flag = True
				
					if not correct_image:
						labels += _line[0]+" "+_line[1]+" "+_line[2]+" "+_line[3]+" "+_line[4]+"\n"

				if save_flag :
					cv2.imwrite("./rotated_image/"+os.path.basename(im_name), img)
					with open (_label_name, "w") as lf:
						lf.writelines(labels)

if __name__=='__main__':
	args = []
	args = sys.argv
	im_dir = args[1]
	rtfl = RotateTrafficLight()
	rtfl.Rotate(im_dir)
