import numpy as np
#import mxnet as mx
import sys
import subprocess
import glob
from mxnet.gluon import nn
from mxnet import image
import mxnet as mx
from collections import namedtuple
import os,time

from mxnet import gluon, nd
from mxnet import autograd as ag
#from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxnet import init

import mxnet.gluon.utils
from mxnet.gluon.model_zoo import vision
from gluoncv.utils import viz
from gluoncv.model_zoo import get_model
from gluoncv.utils import export_block
from train_gluon_mobilev2 import RandomRotateTransform

import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

class SoftMaxBlock(nn.HybridBlock):

	def __init__(self, **kwargs):
		super(SoftMaxBlock, self).__init__(**kwargs)


	def hybrid_forward(self, F, x):
		return F.softmax(x)

class TlrTrainer:
	def __init__(self):
		self.num_epoch = 100
		self.ctx = [mx.cpu()]
		#self.ctx = [mx.cpu()]

	def Train(self):
		classes = []
		result_dir = "results"
		os.makedirs(result_dir)
		with open("classes.names", "r") as fp:
			for line in fp.readlines():
				classes.append(line.rstrip('\n'))
		
		print(classes)
		#classes = ["Anime Person"]
		#net = vision.resnet18_v2(pretrained=True, ctx = self.ctx)

		self.image_shape = 128

		images = glob.glob("/home/tier4-t-iwai/.autoware/tlr_TrainingDataSet/all/*.png")

		ann = {}

		with open("/home/tier4-t-iwai/.autoware/tlr_TrainingDataSet/all/annotation.txt", "r") as fp:
			for line in fp.readlines():
				line_ = line.split(" ")
				if len(line_) == 2:
					ann[line_[0]] = line_[1].rstrip("\n")

		transform_test = transforms.Compose([
			RandomRotateTransform(),
			transforms.Resize(self.image_shape),
			transforms.ToTensor(),
		])

		model_dirs = glob.glob("./models/"+str(self.image_shape)+"/*")
		imgs = {}
		for image_path in images:
			imgs[image_path] = image.imread(image_path)


		for model_dir in model_dirs:
			symbol_name = glob.glob(model_dir+"/*.json")
			param_name = glob.glob(model_dir+"/*.params")
			
			net = gluon.SymbolBlock.imports(symbol_name[0], ['data'], param_name[0])
			all_cnt = 0
			cur_cnt = 0

			with open(result_dir+"/"+model_dir.replace(".","").replace("/", "_")+".txt", "w") as fp:
				start_time = time.time()
				for i in range(5):
					for image_path in images:
						img = transform_test(imgs[image_path])
						inf = net(img.expand_dims(axis = 0))
						inf_sorted = mx.nd.argsort(inf, is_ascend=0)
						name = os.path.basename(image_path)
						all_cnt += 1
						inf_class = classes[int(inf_sorted[0, 0].asscalar())]
						true_or_false = "False"
						if ann[name] in inf_class:
							cur_cnt += 1
							true_or_false = "True"
						fp.write("image_path : "+name+",   result :"+inf_class + " truth : "+ann[name]+" "+true_or_false+ "\n")
					elapsed_time = time.time() - start_time
				fp.write("result : "+str(float(cur_cnt)/all_cnt)+", elapsed_time : "+str(elapsed_time*1000)+"[msec]")
			print(model_dir+", "+str(float(cur_cnt)/all_cnt))


if  __name__=='__main__':
	args = []
	args = sys.argv
	im_dir = None
	tlrTra = TlrTrainer()
	tlrTra.Train()
	#tlrTra.Predict()
	#tlrTra.test()
