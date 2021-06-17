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
import torchtext
from datetime import datetime


## We only train on yes/no questions
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

## Create word frequency dict from vqa data
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

## Question module for handing questions
## Word embedding net with no grad + GRU layers
class QuestionModule(torch.nn.Module):
	def __init__(self, hidden_size, vocab_size, embedding_length, word_embeddings):
		super(QuestionModule, self).__init__()
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length

		# Embedding layers
		self.embeddings = nn.Embedding(self.vocab_size, self.embedding_length)
		self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)
		self.recurrent1 = nn.GRU(self.embedding_length, self.hidden_size)

	def forward(self, question, question_lengths):
		embedded_question = self.embeddings(question).detach()
		packed_question = nn.utils.rnn.pack_padded_sequence(embedded_question, question_lengths, enforce_sorted=False)
		_, hidden_question = self.recurrent1(packed_question)

		return hidden_question
    
## The full network
class Network(torch.nn.Module):
	def __init__(self, img_size, hidden_size, vocab_size, embedding_length, word_embeddings):
		super(Network, self).__init__()
		self.question_module = QuestionModule(hidden_size, vocab_size, embedding_length, word_embeddings)
        
		self.fcs = torch.nn.Sequential(torch.nn.Linear(img_size+hidden_size,5000),
										nn.BatchNorm1d(num_features=5000),
										nn.Dropout(0.2),
										nn.ReLU(),
										nn.Linear(5000, 1000),
										nn.BatchNorm1d(num_features =1000),
										nn.Dropout(0.2),
										nn.ReLU(),
										nn.Linear(1000, 500),
										nn.BatchNorm1d(num_features = 500),
										nn.Dropout(0.2),
										nn.ReLU(),
										nn.Linear(500, 100),
										nn.BatchNorm1d(num_features = 100),
										nn.ReLU(),
										nn.Linear(100, 1))

	def forward(self, question, question_lengths, img):
		x = self.question_module(question, question_lengths)
		x.require_gradient = True
		if len(x.size()) != len(img.size()):
				img = img.unsqueeze(0)
		x = torch.cat((x, img), 2).squeeze(0)
		x = torch.sigmoid(self.fcs(x)).squeeze()

		return x



def train_model(train_dataloader, optimizer, loss, net, validation_loader, is_distributed, use_cuda, log='runs', epochs=5):

	device = torch.device("cuda" if use_cuda else "cpu")
	net.to(device)
	training_losses={}
	training_accuracies={}
	validation_losses={}
	validation_accuracies={}
	print(len(train_dataloader))
	for epoch in range(epochs):        
		running_loss = 0
		correct = 0
		total = 0
		training_losses[epoch+1]=[]
		training_accuracies[epoch+1]=[]
		for i,batch in enumerate(train_dataloader):
			if i % 2000 == 0 and i != 0:
				print('\nAccuracy: '+str(float(correct)/total)+' Loss: '+str(running_loss/total))
			q, q_len = batch.Question
			q = q.to(device)
			img = batch.Image.to(device)
			y = batch.Answer.to(device)
            
			optimizer.zero_grad()

			y_pred = net(q, q_len, img)

			loss_val = loss(y_pred, y)
			loss_val.backward()

			optimizer.step()

			preds = (y_pred>0.5).float()
			correct += (torch.sum(preds==y)).item()
			total += len(y)
			running_loss += loss_val.item()

			training_losses[epoch+1].append(running_loss/total)
			training_accuracies[epoch+1].append(float(correct)/total)
			info = { ('loss') : running_loss/total,('accuracy'): float(correct)/total}


		print('\nEpoch: '+str(epoch+1)+' Accuracy: '+str(float(correct)/total)+' Loss: '+str(running_loss/total))

		# Perform validation testing
		val_correct = 0
		val_total = 0
		val_loss = 0
		with torch.set_grad_enabled(False):
			for batch in validation_loader:
				# Transfer to GPU
				q, q_len = batch.Question
				q = q.to(device)
				img = batch.Image.to(device)
				y = batch.Answer.to(device)
				# Model computations
				y_pred = net(q, q_len, img)

				preds = (y_pred>0.5).float()
				val_correct += (torch.sum(preds==y)).item()
				val_total += len(y)
				loss_val = loss(y_pred, y)
				val_loss += loss_val.item()
                
                
		print('Validation accuracy: '+str(float(val_correct)/val_total))
		validation_losses[epoch+1] = val_loss/val_total
		validation_accuracies[epoch+1] = float(val_correct)/val_total
		torch.save(net.cpu().state_dict(), 'dl_baseline_log/' + str(epoch) +'.pt')
		net = net.to(device)
	if not os.path.exists('.dl_baseline_log'):
		os.makedirs('dl_baseline_log')
	np.save('dl_baseline_log/train_loss.npy', training_losses, allow_pickle=True)
	np.save('dl_baseline_log/train_acc.npy', training_accuracies, allow_pickle=True)
	np.save('dl_baseline_log/val_loss.npy', validation_losses, allow_pickle=True)
	np.save('dl_baseline_log/val_acc.npy', validation_accuracies, allow_pickle=True)


class FinalDataset(Dataset):
	def __init__(self, vqa_data, imgFeatDir, dataSubType):
		self.data = vqa_data
		self.imgFeatDir = imgFeatDir
		self.dataSubType = dataSubType
		return

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):

		q, a, i = self.data[idx]
		imgFilename = 'COCO_' + self.dataSubType + '_'+ str(i).zfill(12) + '.pt'       
		image_vec = torch.load(self.imgFeatDir+'/'+ self.dataSubType + '/'+ imgFilename, map_location=torch.device('cpu')).squeeze(0).squeeze(1).squeeze(1)

		return q[0], torch.tensor([a], dtype=torch.float), image_vec

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
		answerList = ann['answers']

		question = str(self.vqa.getQA(quesId))

		if ann['multiple_choice_answer']=='yes':
			answer = 1
		else:
			answer = 0

		questions = nltk.sent_tokenize(question)
		for i in range(len(questions)):
			questions[i] = questions[i].lower()
			questions[i] = re.sub(r'\W',' ',questions[i])
			questions[i] = re.sub(r'\s+',' ',questions[i])

		return questions, answer, imgId
    
def create_train_loader(feat_dir, is_distributed, **kwargs):

	versionType ='v2_' 
	taskType    ='OpenEnded_' 
	dataType    ='mscoco_'
	dataSubType ='train2014_'

	annFile     ='%s%s%sannotations.json'%(versionType, dataType, dataSubType)
	quesFile    ='%s%s%s%squestions.json'%(versionType, taskType, dataType, dataSubType)

	vqa=VQA(annFile, quesFile)

	annotations = np.load('training_annotations.npy', allow_pickle=True).tolist()

	vqa_data = vqaDataset(vqa, annotations)

	# Insert location of image features
	dataset = FinalDataset(vqa_data, feat_dir,'train2014')

	loader = DataLoader(dataset, batch_size=64)

	return loader

def create_val_loader(feat_dir, **kwargs):

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
	dataset = FinalDataset(vqa_data, feat_dir, 'val2014')

	loader = DataLoader(dataset, batch_size=16, shuffle=True)
	return loader

def get_dataset(loader, fields):
	example_list = []

	for i, (q, a, img) in enumerate(loader):
		for q_i, a_i, img_i in zip(q, a, img):            
			example_list.append(torchtext.data.Example.fromlist([q_i, a_i, img_i.detach().numpy()], fields))

	complete_set = torchtext.data.Dataset(example_list, fields)
	return complete_set

def combine_datasets(train_loader, val_loader):
	TEXT = torchtext.data.Field(sequential=True, lower=True, include_lengths=True)
	ANS = torchtext.data.Field(sequential=False, use_vocab=False, dtype=torch.float, is_target=True)
	IMG = torchtext.data.Field(sequential=False, use_vocab=False, dtype=torch.float)

	fields = [('Question', TEXT),  ('Answer', ANS), ('Image', IMG)]

	train_set = get_dataset(train_loader, fields)
	val_set = get_dataset(val_loader, fields)

	TEXT.build_vocab(train_set, val_set, vectors='glove.6B.300d')

	train_iter = torchtext.data.BucketIterator(
		train_set, 
		batch_size = 16,
		sort_key = lambda x: len(x.Question),
		sort_within_batch = True,
		repeat=False, 
		shuffle=True)

	# Define the test iterator
	val_iter = torchtext.data.BucketIterator(
		val_set, 
		batch_size = 16,
		sort=False,
		sort_key = lambda x: len(x.Question),
		sort_within_batch = False,
		repeat=False, 
		shuffle=False)

	return train_iter, val_iter, len(TEXT.vocab), TEXT.vocab.vectors


def printTime(text):
	now = datetime.now()
	current_time = now.strftime("%H:%M:%S")
	print(current_time, " : " + text, )    
    
if __name__ == '__main__':
	printTime("Training Start.........")

	data_dir = 'Image_Features_ResNet'
	seed = 12423352351
	is_distributed= False
    
	use_cuda=torch.cuda.is_available()

	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
	# set the seed for generating random numbers
	torch.manual_seed(seed)
	if use_cuda:
		torch.cuda.manual_seed(seed)

	train_loader = create_train_loader(data_dir, is_distributed, **kwargs)
	val_loader = create_val_loader(data_dir, **kwargs)

	train_iter, val_iter, vocab_size, word_embeddings = combine_datasets(train_loader, val_loader)

	printTime("Data set created.........")
    
	hidden_size = 1280
	embedding_length = 300
	img_size = 2048

    
	dl_net = Network(img_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    
	printTime("Network created.........")
    
	criterion_log = nn.BCELoss(size_average=True)
    
	parameters = filter(lambda p: p.requires_grad, dl_net.parameters())
	optimizer_log = optim.Adam(parameters,lr = 0.0001)
	epochs = 20
	train_model(train_iter, optimizer_log, criterion_log, dl_net, val_iter, is_distributed, use_cuda, 'dl_baseline', epochs=epochs)
    
	printTime("Finished training.........")
    
	torch.save(dl_net.cpu().state_dict(), 'dl_baseline_log/' + str(epochs) +'.pt')
    
	printTime("Trained Network saved.........")

