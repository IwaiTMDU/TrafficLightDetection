import numpy as np
import mxnet as mx
import sys
import subprocess
import glob
from collections import namedtuple
import os

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn

import mxnet.gluon.utils
from mxnet.gluon.model_zoo import vision

import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

class TlrTrainer:
	def __init__(self):
		self.num_epoch = 50
		self.ctx = mx.gpu(0)

	def Train(self, _image_dir):

		#net = vision.resnet18_v2(pretrained=True, ctx = self.ctx)

		train_data = mx.io.ImageRecordIter(
			path_imgrec = _image_dir+'/train/image.rec',
			path_imgidx = _image_dir+'/train/image.idx',
			shuffle=True,
			batch_size = 32,
			data_shape=(3, 224, 224),
			rand_mirror = True,
			random_resized_crop = True,
		)

		val_data = mx.io.ImageRecordIter(
			path_imgrec = _image_dir+'/val/image.rec',
			path_imgidx = _image_dir+'/val/image.idx',
			shuffle = False,
			batch_size = 32,
			data_shape = (3, 224, 224),
		)

		print("Loading Resnet18")

		symbol, arg_params, aux_params = mx.model.load_checkpoint('resnet-18', 0)
		print(symbol.list_arguments())
		
		all_layers = symbol.get_internals()
		net = all_layers['flatten0'+'_output']
		net = mx.symbol.FullyConnected(data=net, num_hidden = 4, name='fc1')
		net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
		args = dict({k:arg_params[k] for k in arg_params if 'fc1' not in k})

		mod = mx.mod.Module(symbol=net, context = self.ctx)
		subprocess.call(
			"rm -rf ./backup ; mkdir -p ./backup", shell=True)
		checkpoint = mx.callback.do_checkpoint("./backup/tlr_resnet18", period = 10)
		mod.fit(train_data, val_data, num_epoch = self.num_epoch, arg_params = args, aux_params=aux_params, epoch_end_callback=checkpoint, allow_missing=True, batch_end_callback = mx.callback.Speedometer(32, 10), kvstore='device', optimizer='sgd', optimizer_params={'learning_rate':0.01},initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2), eval_metric='acc')
		metric = mx.metric.Accuracy()
		mod_score = mod.score(val_data, metric)
		assert mod_score > 0.77, "Low training accuracy."
		print(mod_score)
		
	
		
	def test(self):
		symbol, arg_params, aux_params = mx.model.load_checkpoint('tlr_resnet18', 100)
		mod = mx.mod.Module(symbol=symbol, context = self.ctx)
		mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], label_shapes=mod._label_shapes)
		mod.set_params(arg_params, aux_params)
		im_list = glob.glob("./test_tl/*.jpg")

		im_lst_list = []

		with open("./test_tl/image.lst") as f:
			for line in f:
				im_lst_list.append(line)


		Batch = namedtuple('Batch', ['data'])
		classes = ["Red", "Yellow", "Green", "Undefined"]
		for _im in im_list:
			print(_im)
			img = mx.image.imread(_im)
			img = img.transpose((2, 0, 1))
			img = img.expand_dims(axis=0)
			img = img.astype('float32')

			arg = Batch([img])
			mod.forward(arg)
			
			prob = mod.get_outputs()[0].asnumpy()
			prob = np.squeeze(prob)
			a = np.argsort(prob)[::-1]

			max_prob = -0.1
			index = 0
			for i in a[0:4]:
				if(max_prob < prob[i]):
					max_prob = prob[i]
					index = i
			
			correct_count = 0
			err_count = 0

			for i, st in enumerate(im_lst_list):
				if os.path.basename(_im) in st:
					c_class = int(st.split()[1])
					if c_class == index:
						correct_count += 1
					else:
						err_count+=1
					im_lst_list.pop(i)
					break

		print("err = "+str(float(err_count)*100.0/(correct_count+err_count)))

	def Predict(self):
		symbol, arg_params, aux_params = mx.model.load_checkpoint('tlr_resnet18', 100)
		mod = mx.mod.Module(symbol=symbol, context = self.ctx)
		mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], label_shapes=mod._label_shapes)
		mod.set_params(arg_params, aux_params)
		im_list = glob.glob("./extracted_tl_image/val/*.jpg")
		Batch = namedtuple('Batch', ['data'])
		classes = ["Red", "Yellow", "Green", "Undefined"]
		for _im in im_list:
			print(_im)
			img = mx.image.imread(_im)
			img = img.transpose((2, 0, 1))
			img = img.expand_dims(axis=0)
			img = img.astype('float32')

			arg = Batch([img])
			mod.forward(arg)
			
			prob = mod.get_outputs()[0].asnumpy()
			prob = np.squeeze(prob)
			a = np.argsort(prob)[::-1]

			max_prob = -0.1
			index = 0
			for i in a[0:4]:
				if(max_prob < prob[i]):
					max_prob = prob[i]
					index = i
			print('probability=%f, class = %s' %(prob[index], classes[index]))
			print("\n")
			

if  __name__=='__main__':
	args = []
	args = sys.argv
	im_dir = args[1]
	tlrTra = TlrTrainer()
	#tlrTra.Train(im_dir)
	#tlrTra.Predict()
	tlrTra.test()