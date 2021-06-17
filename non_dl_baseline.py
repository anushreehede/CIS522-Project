# -*- coding: utf-8 -*-
"""
non-DL-baseline.ipynb

"""


from vqa_py3 import VQA 
import random
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

def create_word_freqs(vqa_data):
	wordfreq = {}
	for (q, a, i) in vqa_data:
		for sentence in q:
			tokens = nltk.word_tokenize(sentence)
			for token in tokens:
				if token not in wordfreq.keys():
					wordfreq[token] = 1
				else:
					wordfreq[token] += 1
	return wordfreq

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
		
		question_tokens = nltk.word_tokenize(q[0])
		ques_vec = []
		for token in self.wordfreq:
			if token in question_tokens:
				ques_vec.append(1)
			else:
				ques_vec.append(0)
		ques_vec = torch.tensor(ques_vec, dtype=torch.float)
		imgFilename = 'COCO_' + self.dataSubType + '_'+ str(i).zfill(12) + '.pt'       
		image_vec = torch.load(self.imgFeatDir+'/'+ self.dataSubType + '/'+ imgFilename, map_location=torch.device('cpu')).squeeze(0).squeeze(1).squeeze(1)

		return ques_vec, torch.tensor([a], dtype=torch.float), image_vec

class LogisticRegression(torch.nn.Module):
	def __init__(self, img_feat_size, text_feat_size):
		super(LogisticRegression, self).__init__()
		self.fc1 = nn.Linear(img_feat_size + text_feat_size, 1)
		self.sigmoid = nn.Sigmoid()
	def forward(self,x):
		outputs = self.sigmoid(self.fc1(x))
		return outputs

def _average_gradients(model):
	# Gradient averaging.
	size = float(dist.get_world_size())
	for param in model.parameters():
		dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
		param.grad.data /= size

def train_model(train_dataloader, optimizer, loss, net, validation_loader, is_distributed, use_cuda, log='runs', epochs=5):
	device = torch.device("cuda" if use_cuda else "cpu")
	net.to(device)
	training_losses={}
	training_accuracies={}
	validation_losses={}
	validation_accuracies={}
	for epoch in range(epochs):
		running_loss = 0
		correct = 0
		total = 0
		training_losses[epoch+1]=[]
		training_accuracies[epoch+1]=[]
		for i, (q, a, img) in enumerate(train_dataloader):

			x = torch.cat((q, img), 1).to(device)
			y = a.to(device)

			optimizer.zero_grad()

			y_pred = net(x)
            
			loss_val = loss(y_pred, y)
			loss_val.backward()

			optimizer.step()

			preds = (y_pred>0.5).float()
			correct += (torch.sum(preds==y)).item()
			total += len(y)
			running_loss += loss_val.item()

			training_losses[epoch+1].append(running_loss/total)
			training_accuracies[epoch+1].append(float(correct)/total)

		print('\nEpoch: '+str(epoch+1)+' Accuracy: '+str(float(correct)/total)+' Loss: '+str(running_loss/total))

		# Perform validation testing
		val_correct = 0
		val_total = 0
		val_loss = 0
		with torch.set_grad_enabled(False):
			for i, (q, a, img) in enumerate(validation_loader):
				# Transfer to GPU
				x = torch.cat((q, img), 1).to(device)
				y = a.to(device)
				# Model computations
				y_pred = net(x)
				preds = (y_pred>0.5).float()
				val_correct += (torch.sum(preds==y)).item()
				val_total += len(y)
				loss_val = loss(y_pred, y)
				val_loss += loss_val.item()

		print('Validation accuracy: '+str(float(val_correct)/val_total)) 
		validation_losses[epoch+1] = val_loss/val_total
		validation_accuracies[epoch+1] = float(val_correct)/val_total
	if not os.path.exists('nondl_baseline_log'):
		os.makedirs('nondl_baseline_log')
	np.save('nondl_baseline_log/non_dl_train_loss.npy', training_losses, allow_pickle=True)
	np.save('nondl_baseline_log/non_dl_train_acc.npy', training_accuracies, allow_pickle=True)
	np.save('nondl_baseline_log/non_dl_val_loss.npy', validation_losses, allow_pickle=True)
	np.save('nondl_baseline_log/non_dl_val_acc.npy', validation_accuracies, allow_pickle=True)

def create_train_loader(feat_dir, is_distributed):

	versionType ='v2_' 
	taskType    ='OpenEnded_' 
	dataType    ='mscoco_'
	dataSubType ='train2014_'

	annFile     ='%s%s%sannotations.json'%(versionType, dataType, dataSubType)
	quesFile    ='%s%s%s%squestions.json'%(versionType, taskType, dataType, dataSubType)


	# initialize VQA api for QA annotations
	vqa=VQA(annFile, quesFile)

	annotations = np.load('training_annotations.npy', allow_pickle=True).tolist()

	vqa_data = vqaDataset(vqa, annotations)

	wordfreq = create_word_freqs(vqa_data)

	most_freq = heapq.nlargest(1024, wordfreq, key=wordfreq.get)
	# Insert location of image features
	dataset = FinalDataset(vqa_data, most_freq, feat_dir,'train2014')

	loader = DataLoader(dataset, batch_size=16)

	return loader, most_freq

def create_val_loader(feat_dir, freq):

	versionType ='v2_' 
	taskType    ='OpenEnded_' 
	dataType    ='mscoco_'
	dataSubType ='val2014_'

	annFile     ='%s%s%sannotations.json'%(versionType, dataType, dataSubType)
	quesFile    ='%s%s%s%squestions.json'%(versionType, taskType, dataType, dataSubType)

	vqa=VQA(annFile, quesFile)
	# initialize VQA api for QA annotations
	annotations = filter_yes_no(vqa, dataSubType, bad_ids=np.load('bad_im_files_val.npy').tolist())

	vqa_data = vqaDataset(vqa, annotations)

	# Insert location of image features
	dataset = FinalDataset(vqa_data, freq, feat_dir, 'val2014')

	loader = DataLoader(dataset, batch_size=1, shuffle=True)
	return loader


if __name__ == '__main__':

	data_dir = 'Image_Features_ResNet'
	seed = 12423352351
	is_distributed= False
    
	use_cuda=torch.cuda.is_available()

	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
	# set the seed for generating random numbers
	torch.manual_seed(seed)
	if use_cuda:
		torch.cuda.manual_seed(seed)

	train_loader, most_freq = create_train_loader(data_dir, is_distributed)
	val_loader = create_val_loader(data_dir, most_freq)

	np.save('bag_of_words.npy', most_freq, allow_pickle=True)

	model_log = LogisticRegression(2048, 1024)

	criterion_log = nn.BCELoss(size_average=True)
	optimizer_log = optim.Adam(model_log.parameters(),lr = 0.0001)

	train_model(train_loader, optimizer_log, criterion_log, model_log, val_loader, is_distributed, use_cuda, 'simple_baseline', epochs=20)

	torch.save(model_log.cpu().state_dict(), 'nondl_baseline_log/nondl_basline.pt')

