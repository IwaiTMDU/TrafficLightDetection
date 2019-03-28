import numpy as np
import mxnet as mx
import sys
import subprocess
import glob
from collections import namedtuple
import os,time

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxnet import init

import mxnet.gluon.utils
from mxnet.gluon.model_zoo import vision
from gluoncv.utils import viz
from gluoncv.model_zoo import get_model
from gluoncv.utils import export_block

import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

class TlrTrainer:
	def __init__(self):
		self.num_epoch = 100
		#self.ctx = [mx.gpu(0)]
		self.ctx = [mx.cpu()]

	def Train(self, _image_dir):
		classes = ["red", "yellow", "green", "undefined"]

		print(classes)
		#classes = ["Anime Person"]
		#net = vision.resnet18_v2(pretrained=True, ctx = self.ctx)

		model_name = "mobilenetv2_0.25"
		BATCH_SIZE = 8
		NUM_WORKERES  = 8

		print("Loading "+model_name)

		finetune_net = get_model(model_name, pretrained = True)

		transform_train = transforms.Compose([
			transforms.Resize(224),
			transforms.RandomFlipLeftRight(),
			transforms.RandomContrast(contrast = 0.5),
			transforms.ToTensor(),
		])

		transform_test = transforms.Compose([
			transforms.Resize(224),
			transforms.ToTensor(),
			
		])

		train_data = gluon.data.DataLoader(gluon.data.vision.ImageRecordDataset("./extracted_tl_image/train/image.rec").transform_first(transform_train),batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERES)
		val_data = gluon.data.DataLoader(gluon.data.vision.ImageRecordDataset("./extracted_tl_image/val/image.rec").transform_first(transform_test),batch_size=BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERES)
		
		finetune_net.hybridize(static_alloc=True,static_shape=True)
		
		with finetune_net.name_scope():
			finetune_net.output = nn.HybridSequential(prefix='output_')
			with finetune_net.output.name_scope():
				finetune_net.output.add(nn.Conv2D(len(classes), 1, use_bias = False, prefix = 'pred_'), nn.Flatten())
		

		finetune_net.initialize(init.Xavier(), ctx = self.ctx)
		finetune_net.collect_params().reset_ctx(self.ctx)
		
		lr = 0.1
		lr_decay_epoch = [30, 60, 90, np.inf]

		trainer = gluon.Trainer(finetune_net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum':0.0005, 'wd':0.0001})

		metric = mx.metric.Accuracy()
		L = gluon.loss.SoftmaxCrossEntropyLoss()

		num_batch = len(train_data)
		param_name = ""

		subprocess.call("rm -rf ./backup ; mkdir -p ./backup", shell=True)

		ssd_model_name = model_name+"_tlr"

		lr_decay_count = 0
		for epoch in range(1, self.num_epoch+1):
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
					outputs = [finetune_net(X) for X in data]
					loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
				for l in loss:
					l.backward()
				
				trainer.step(BATCH_SIZE)
				train_loss += sum([l.mean().asscalar() for l in loss])/len(loss)

				metric.update(label, outputs)

			_, train_acc = metric.get()
			train_loss /= num_batch

			_, val_acc = self.test(finetune_net, val_data, self.ctx)
			print('[Epoch %d] Train-acc: %.3f, loss: %.3f | Val-acc: %.3f | time: %.1f' %(epoch, train_acc, train_loss, val_acc, time.time() - tic))
			finetune_net.save_parameters(param_name)

		finetune_net.collect_params().reset_ctx(mx.cpu())
		export_block(ssd_model_name+'_tlr', finetune_net, data_shape=(224, 224, 3))
		#symbol, arg_params, aux_params = mx.model.load_checkpoint(model_name+'_knmz', 0)
		#symbol = mx.symbol.SoftmaxOutput(data=symbol, name='softmax')
		#symbol.tojson()
		#symbol.save(model_name+'_knmz-symbol.json')

	def test(self, net, val_data, ctx):
		metric = mx.metric.Accuracy()
		for i, batch in enumerate(val_data):
			data = gluon.utils.split_and_load(batch[0], ctx_list = ctx, batch_axis = 0, even_split=False)
			label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis = 0, even_split = False)
			outputs = [net(X) for X in data]
			metric.update(label, outputs)

		return metric.get()

	def Predict(self):
		pass
			

if  __name__=='__main__':
	args = []
	args = sys.argv
	im_dir = args[1]
	tlrTra = TlrTrainer()
	tlrTra.Train(im_dir)
	#tlrTra.Predict()
	#tlrTra.test()