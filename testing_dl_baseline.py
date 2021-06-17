# -*- coding: utf-8 -*-
"""
non-DL-baseline.ipynb

"""

from vqa_py3 import VQA 
import random
import dill

import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader

import nltk
nltk.download('punkt')
import re
import heapq
import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.distributed as dist
import argparse
import skimage.io as io
import simplejson as json
import logging as logger

from textwrap import wrap
import re
import itertools
import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix
import torchtext

##### Comment out the ones not being used. 

from dl_baseline import QuestionModule, Network 


def filter_yes_no(vqa, subtype, bad_ids):
	quesIds = vqa.getQuesIds()
	imgIds = vqa.getImgIds()

	annotations = []
	for idx in range(len(imgIds)):
		ann = vqa.loadQA([quesIds[idx]])[0]
		imgFilename = 'COCO_' + subtype + str(ann['image_id']).zfill(12)+'.jpg'
		if ann['answer_type'] == 'yes/no' and imgFilename not in bad_ids:
			annotations.append(ann)
	return annotations

class vqaDataset(Dataset):
	def __init__(self, vqa, annotations):
		self.imgIds = vqa.getImgIds()
		self.quesIds = vqa.getQuesIds()
		self.annotations = annotations
		self.vqa = vqa

		return

	def __len__(self):
		return len(self.annotations)

	def __getitem__(self, idx):
		
		ann = self.annotations[idx]

		imgId = ann['image_id']
		quesId = ann['question_id']

		question = str(self.vqa.getQA(quesId))
		if ann['multiple_choice_answer'] == 'yes':
			answer = 1
		else:
			answer = 0 

		questions = nltk.sent_tokenize(question)
		for i in range(len(questions)):
			questions[i] = questions[i].lower()
			questions[i] = re.sub(r'\W',' ',questions[i])
			questions[i] = re.sub(r'\s+',' ',questions[i])

		return questions, answer, imgId

class FinalDataset(Dataset):
	def __init__(self, vqa_data, wordfreq, imgFeatDir, dataSubType):
		self.data = vqa_data
		self.wordfreq = wordfreq
		self.imgFeatDir = imgFeatDir
		self.dataSubType = dataSubType
		return

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		
		q, a, i = self.data[idx]
		
		if len(self.wordfreq) != 0:
			question_tokens = nltk.word_tokenize(q[0])
			ques_vec = []
			for token in self.wordfreq:
				if token in question_tokens:
					ques_vec.append(1)
				else:
					ques_vec.append(0)
			ques_vec = torch.tensor(ques_vec, dtype=torch.float)
		else:
			ques_vec = q[0]
		imgFilename = 'COCO_' + self.dataSubType + '_'+ str(i).zfill(12) + '.pt'       

		image_vec = torch.load(self.imgFeatDir+'/'+ self.dataSubType + '/'+ imgFilename, map_location=torch.device('cpu')).squeeze(0).squeeze(1).squeeze(1)

		return ques_vec, torch.tensor([a], dtype=torch.float), image_vec

def test_model(test_loader, loss, net, use_cuda):

	device = torch.device("cuda" if use_cuda else "cpu")
	net.to(device)

	# Perform validation testing
	test_correct = 0
	test_total = 0
	test_loss = 0
	true_answers = []
	predicted_answers = []
	with torch.set_grad_enabled(False):
		for batch in test_loader:
			# Transfer to GPU
			q, q_len = batch.Question
			q = q.to(device)
			img = batch.Image.to(device)
			y = batch.Answer.to(device)
			# Model computations
			y_pred = net(q, q_len, img)
			preds = (y_pred>0.5).float()
			test_correct += (torch.sum(preds==y)).item()
			test_total += len(y)
			loss_val = loss(y_pred, y)
			test_loss += loss_val.item()

			# Code to store predictions
			true_answers.append(y.item())
			predicted_answers.append(preds.item())

	print('Test accuracy: '+str(float(test_correct)/test_total))
	return true_answers, predicted_answers 

def create_test_loader(feat_dir, freq={}):

	versionType ='v2_' 
	taskType    ='OpenEnded_' 
	dataType    ='mscoco_'
	dataSubType ='val2014_'

	annFile     ='%s%s%sannotations.json'%(versionType, dataType, dataSubType)
	quesFile    ='%s%s%s%squestions.json'%(versionType, taskType, dataType, dataSubType)

	vqa=VQA(annFile, quesFile)
	annotations = filter_yes_no(vqa, dataSubType, bad_ids=np.load('bad_im_files_val.npy').tolist())
	
	vqa_data = vqaDataset(vqa, annotations)


	# Insert location of image features
	dataset = FinalDataset(vqa_data, freq, feat_dir, 'val2014')

	# print(dataset[300][0])

	loader = DataLoader(dataset, batch_size=1, shuffle=False)
	return loader, vqa, annotations 
# 	return None, None, None

def get_dataset(loader, fields):
	example_list = []

	for i, (q, a, img) in enumerate(loader):
		for q_i, a_i, img_i in zip(q, a, img):            
			example_list.append(torchtext.data.Example.fromlist([q_i, a_i, img_i.detach().numpy()], fields))

	complete_set = torchtext.data.Dataset(example_list, fields)
	return complete_set

def combine_datasets(loader):
	# TEXT = torchtext.data.Field(sequential=True, lower=True, include_lengths=True)
	TEXT = dill.load(open('train_val_vocab_baseline', 'rb'))
	ANS = torchtext.data.Field(sequential=False, use_vocab=False, dtype=torch.float, is_target=True)
	IMG = torchtext.data.Field(sequential=False, use_vocab=False, dtype=torch.float)

	fields = [('Question', TEXT),  ('Answer', ANS), ('Image', IMG)]

	test_set = get_dataset(loader, fields)


	# Define the test iterator
	test_iter = torchtext.data.BucketIterator(
		test_set, 
		batch_size = 1,
		sort=False,
		sort_key = lambda x: len(x.Question),
		sort_within_batch = False,
		repeat=False, 
		shuffle=False)
	
	return test_iter, len(TEXT.vocab), TEXT.vocab.vectors

# Credits - https://stackoverflow.com/questions/41617463/tensorflow-confusion-matrix-in-tensorboard
def plot_confusion_matrix(correct_labels, predict_labels, labels, display_labels, title='Confusion matrix', tensor_name = 'MyFigure/image', normalize=False):
	''' 
	Parameters:
	  correct_labels                  : These are your true classification categories.
	  predict_labels                  : These are you predicted classification categories
	  labels                          : This is a lit of labels which will be used to display the axis labels
	  title='Confusion matrix'        : Title for your matrix
	  tensor_name = 'MyFigure/image'  : Name for the output summay tensor

	Returns:
	  summary: TensorFlow summary 

	Other itema to note:
	  - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc. 
	  - Currently, some of the ticks dont line up due to rotations.
	'''
	cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
	if normalize:
		cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
		cm = np.nan_to_num(cm, copy=True)
		cm = cm.astype('int')

	np.set_printoptions(precision=2)
	###fig, ax = matplotlib.figure.Figure()

	fig = matplotlib.pyplot.figure(figsize=(2, 2), dpi=320, facecolor='w', edgecolor='k')
	ax = fig.add_subplot(1, 1, 1)
	im = ax.imshow(cm, cmap='Oranges')

	classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in display_labels]
	classes = ['\n'.join(wrap(l, 40)) for l in classes]

	tick_marks = np.arange(len(classes))

	ax.set_xlabel('Predicted', fontsize=7)
	ax.set_xticks(tick_marks)
	c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
	ax.xaxis.set_label_position('bottom')
	ax.xaxis.tick_bottom()

	ax.set_ylabel('True Label', fontsize=7)
	ax.set_yticks(tick_marks)
	ax.set_yticklabels(classes, fontsize=4, va ='center')
	ax.yaxis.set_label_position('left')
	ax.yaxis.tick_left()

	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
	fig.set_tight_layout(True)
	matplotlib.pyplot.show()

	return

def main(model_name):

	data_dir = 'Image_Features_ResNet'
	seed = 12423352351
	is_distributed= False
    
	use_cuda=torch.cuda.is_available()

	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
	# set the seed for generating random numbers
	torch.manual_seed(seed)
	if use_cuda:
		torch.cuda.manual_seed(seed)


	most_freq = {}

	test_loader, vqa, annotations = create_test_loader(data_dir, most_freq)

	hidden_size = 1280
	embedding_length = 300
	img_size = 2048
	test_loader, vocab_size, word_embeddings = combine_datasets(test_loader)
	model = Network(img_size, hidden_size, vocab_size, embedding_length, word_embeddings) # All DL-Models. Need to import correctly.
    
	model.load_state_dict(torch.load(model_name))
	model.eval()
	
	criterion_log = nn.BCELoss(size_average=True)
	# optimizer_log = optim.Adam(model_log.parameters(),lr = 0.0001)

	true, pred = test_model(test_loader, criterion_log, model, use_cuda)

	# Confusion matrix
	plot_confusion_matrix(true, pred, [0, 1], ["no", "yes"])

	for i in range(5):
		n = random.randint(0, len(pred)-1)
		print(annotations[n]['question_id'])
		print(vqa.getQA(annotations[n]['question_id']))
		print('actual: '+annotations[n]['multiple_choice_answer'])

		if pred[n]==0:
			print('predicted: '+"no")
		if pred[n]==1:
			print('predicted: '+"yes")

if __name__ == '__main__':

	model_name = 'fixed_dl_baseline_/6.pt'

    
	main(model_name)
