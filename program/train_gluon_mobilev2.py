import numpy as np
#import mxnet as mx
import sys
import subprocess
import glob
from mxnet.gluon import nn
import mxnet as mx
from collections import namedtuple
import os,time
import random

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
from gluoncv.model_zoo.densenet import get_densenet

from augumentation.rotate_traffic_light import RotateTrafficLight

import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

class SoftMaxBlock(nn.HybridBlock):

	def __init__(self, **kwargs):
		super(SoftMaxBlock, self).__init__(**kwargs)


	def hybrid_forward(self, F, x):
		return F.softmax(x)

class RandomRotateTransform(nn.Block):
	def forward(self,x):
		ret = random.randint(0,1)
		if ret == 0:
			np_image = x.asnumpy()
			Rot = RotateTrafficLight()
			np_image,_ = Rot.RotateImage(np_image)
			return mx.nd.array(np_image)
		else:
			return x

class TlrTrainer:
	def __init__(self):
		self.num_epoch = 300
		self.ctx = [mx.gpu(0)]
		#self.ctx = [mx.cpu()]

	def Train(self, _image_dir):
		classes = []
		with open("classes.names", "r") as fp:
			for line in fp.readlines():
				classes.append(line.rstrip('\n'))

		print(classes)
		#classes = ["Anime Person"]
		#net = vision.resnet18_v2(pretrained=True, ctx = self.ctx)

		model_name = "mobilenetv2_1.0"
		BATCH_SIZE = 32
		self.image_shape = 128
		NUM_WORKERES  = 16  

		print("Loading "+model_name)

		mobilenet = True

		transform_train = transforms.Compose([
			transforms.RandomFlipTopBottom(),
			transforms.RandomBrightness(0.5),
			transforms.RandomContrast(contrast = 0.5),
			transforms.RandomSaturation(0.5),
			transforms.RandomFlipTopBottom(),
			RandomRotateTransform(),
			transforms.Resize(self.image_shape),
			transforms.ToTensor(),
		])

		transform_test = transforms.Compose([
			transforms.Resize(self.image_shape),
			transforms.ToTensor(),
		])

		train_data = gluon.data.DataLoader(gluon.data.vision.ImageRecordDataset("./tlr_dataset_train.rec").transform_first(transform_train),batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERES)
		val_data = gluon.data.DataLoader(gluon.data.vision.ImageRecordDataset("./tlr_dataset_val.rec").transform_first(transform_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERES)
		
		if mobilenet:
			self.net = get_model(model_name, pretrained=False)
			self.net.hybridize(static_alloc=True,static_shape=True)
			with self.net.name_scope():
				self.net.output = nn.HybridSequential(prefix='output_')
				with self.net.output.name_scope():
					self.net.output.add(nn.Conv2D(len(classes), 1, use_bias = False, prefix = 'pred_'), nn.Flatten(), SoftMaxBlock())
		else:
			self.net = get_densenet(51, pretrained=False, ctx=mx.gpu(0))
		

		self.net.initialize(init.Xavier(), ctx = self.ctx)
		'''
		if _image_dir is not None:
			self.net.load_parameters(
				_image_dir, self.ctx, allow_missing=True, ignore_extra=True)
		'''
		self.net.collect_params().reset_ctx(self.ctx)
		
		lr = 0.01
		lr_decay_epoch = [30, 60, 90, np.inf]

		trainer = gluon.Trainer(self.net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum':0.0005})
		#trainer = gluon.Trainer(self.net.collect_params(), 'adam', {'learning_rate': 0.0002})
		metric = mx.metric.Accuracy()
		L = gluon.loss.SoftmaxCrossEntropyLoss(from_logits = True)

		num_batch = len(train_data)
		param_name = ""

		subprocess.call("rm -rf ./backup ; mkdir -p ./backup", shell=True)

		ssd_model_name = model_name+"_tlr"

		lr_decay_count = 0
		_epoch = 0
		try:
			for epoch in range(1, self.num_epoch+1):
				_epoch = epoch
				param_name = "./backup/"+ssd_model_name+"_"+str(epoch)+".param"

				tic = time.time()
				train_loss = 0
				metric.reset()

				if epoch == lr_decay_epoch[lr_decay_count]:
					trainer.set_learning_rate(trainer.learning_rate*0.1)
					lr_decay_count += 1
					print("Refine lr : " + str(trainer.learning_rate))

				for i, batch in enumerate(train_data):
					
					data  = gluon.utils.split_and_load(batch[0], ctx_list = self.ctx, batch_axis = 0, even_split = False)
					label = gluon.utils.split_and_load(batch[1], ctx_list = self.ctx, batch_axis = 0, even_split = False)
					with ag.record():
						outputs = [mx.nd.log(self.net(X)) for X in data]
						loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
					for l in loss:
						l.backward()
					
					trainer.step(BATCH_SIZE)
					train_loss += sum([l.mean().asscalar() for l in loss])/len(loss)

					metric.update(label, outputs)

				_, train_acc = metric.get()
				train_loss /= num_batch

				_, val_acc = self.test(self.net, val_data, self.ctx)
				print('[Epoch %d] Train-acc: %.3f, loss: %.3f | Val-acc: %.3f | time: %.1f' %(epoch, train_acc, train_loss, val_acc, time.time() - tic))
				self.net.save_parameters(param_name)
		except KeyboardInterrupt:
			print("Keyboard interrupted")
			self.WriteJson()
			sys.exit

		self.WriteJson()
		#self.net.collect_params().reset_ctx(mx.cpu())
		#export_block("backup/mxnet-network", self.net,
        #            data_shape=(self.image_shape, self.image_shape, 3), layout="CHW", preprocess=False)


	def test(self, net, val_data, ctx):
		metric = mx.metric.Accuracy()
		for i, batch in enumerate(val_data):
			data = gluon.utils.split_and_load(batch[0], ctx_list = ctx, batch_axis = 0, even_split=False)
			label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis = 0, even_split = False)
			outputs = [net(X) for X in data]
			metric.update(label, outputs)

		return metric.get()

	def WriteJson(self):
		self.net.collect_params().reset_ctx(mx.cpu())
		
		export_block("backup/mxnet_network", self.net, data_shape=(
			self.image_shape, self.image_shape, 3), layout="CHW", preprocess=False)
		self.net.collect_params().reset_ctx(mx.gpu(0))
		#symbol, arg_params, aux_params = mx.model.load_checkpoint("mxnet_network", 0)

if  __name__=='__main__':
	args = []
	args = sys.argv
	im_dir = None
	if len(args) > 1:
		im_dir = args[1]
	tlrTra = TlrTrainer()
	tlrTra.Train(im_dir)
	#tlrTra.Predict()
	#tlrTra.test()
