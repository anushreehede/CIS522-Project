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
import time
import torch.nn as nn 
import torch.optim as optim
import torch.distributed as dist
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
import skimage.io as io
import simplejson as json
import logging as logger
import torchtext
from datetime import datetime

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

    
class QuestionModule(torch.nn.Module):
	def __init__(self, hidden_size, vocab_size, embedding_length, word_embeddings):
		super(QuestionModule, self).__init__()
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length

		# Embedding layers
		self.embeddings = nn.Embedding(self.vocab_size, self.embedding_length)
		self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)
        
# 		self.embeddings.weight = nn.Parameter(word_embeddings)
		self.recurrent1 = nn.GRU(self.embedding_length, self.hidden_size)

	def forward(self, question, question_lengths):
# 		print('reached q module', question.shape, question_lengths.shape)
		embedded_question = self.embeddings(question).detach()
		packed_question = nn.utils.rnn.pack_padded_sequence(embedded_question, question_lengths, enforce_sorted=False)
		_, hidden_question = self.recurrent1(packed_question)

		return hidden_question
    
class FCNet(nn.Module):
    """
    Fully-connected network used in Attention module
    """
    def __init__(self, input_dim, output_dim):
        super(FCNet, self).__init__()
        self.fc1 = weight_norm(nn.Linear(input_dim, output_dim), dim = None)
        self.relu = nn.LeakyReLU(.1)
        self.dropout = nn.Dropout(0.2)
        
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x 

class AttentionModule(nn.Module):
    def __init__(self, embed_dim, feat_dim, num_hid, dropout):
        """
        embed_dim = dimension of gru output
        feat_dim = dimension of image featrues
        num_hid = dimension of hidden features in attention net
        """
        super(AttentionModule, self).__init__()
        
        self.ques_net = FCNet(embed_dim, num_hid)
        self.feat_net = FCNet(feat_dim, num_hid)
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim = None)
        
    def forward(self, feat, embed):
        """
        feat: image feature output [batch, k, img_features]
        embed: gru output [batch, hidden dim]
        """
        logits = self.logits(feat, embed)
        output = F.softmax(logits, 1)
        return output
    
    def logits(self, feat, embed):
        """
        feat: image feature output [batch, k, img_features]
        embed: gru output [batch, hidden dim]
        """
        batch, k, _  = feat.size()
        feat_embed = self.feat_net(feat) # [batch, k, embed dim]
        ques_embed = self.ques_net(embed).unsqueeze(1).repeat(1,k,1)
        weighted = feat_embed * ques_embed
        weighted = self.dropout(weighted)
        logits = self.linear(weighted)
        return logits
    
    
class Network(torch.nn.Module):
	def __init__(self, img_size, hidden_size, vocab_size, embedding_length, word_embeddings):
		super(Network, self).__init__()
		self.question_module = QuestionModule(hidden_size, vocab_size, embedding_length, word_embeddings)
		self.question_attention_module = AttentionModule(embed_dim = 1280, feat_dim = 2048, num_hid = 1024, dropout = 0.2)
		self.object_attention_module = AttentionModule(embed_dim = 400, feat_dim = 2048, num_hid = 1024, dropout = 0.2)       
		self.question_fc = FCNet(1280, 1280) #https://arxiv.org/pdf/1707.07998.pdf
		self.final_attended_fc = FCNet(2048, 1280)
		self.object_fc = FCNet(400, 1280)
		self.fcs = torch.nn.Sequential(torch.nn.Linear(1280,5000),
										nn.BatchNorm1d(num_features=5000),
										nn.Dropout(0.2),
										nn.LeakyReLU(.1),
										nn.Linear(5000, 1000),
										nn.BatchNorm1d(num_features =1000),
										nn.Dropout(0.2),
										nn.LeakyReLU(.1),
										nn.Linear(1000, 500),
										nn.BatchNorm1d(num_features = 500),
										nn.Dropout(0.2),
										nn.LeakyReLU(.1),
										nn.Linear(500, 100),
										nn.BatchNorm1d(num_features = 100),
										nn.LeakyReLU(.1),
										nn.Linear(100, 1))

	def forward(self, question, question_lengths, img_feat, obj_feat):
        
        
		#Consider removing the question embedding fc net and trying leaky relus
        
		#Generate GRU question embedding
		ques_embed = self.question_module(question, question_lengths)
		ques_embed.require_gradient = True
		ques_embed = torch.squeeze(ques_embed, 0) #[batch, 1280]
        
		#Generate question attended image features
		ques_att_weights = self.question_attention_module(img_feat, ques_embed) #[batch, k, 1]
		attended_img_feat = (ques_att_weights * img_feat)#Apply attention to the image features [batch, k, 2048]
   
		#Take obj_feat and pass through attention module with attended im feat
		obj_att_weights = self.object_attention_module(attended_img_feat, obj_feat) #[batch, k, 1]
		double_attended_img_feat = (obj_att_weights * attended_img_feat).sum(1) #[batch, 2048]
        
		#Pass both final attended img feat and obj feat through fc and sum
		final_att_fc_out = self.final_attended_fc(double_attended_img_feat) #[batch, 1280]
		final_obj_fc_out = self.object_fc(obj_feat) #[batch, 1280]
		pre_mult_img_feat = final_att_fc_out + final_obj_fc_out #[batch, 1280]
        
		#Pointwise multiply question embedding with summed attended features and obj features
		final_ques_embed = self.question_fc(ques_embed) #might not need this
		final_feats = final_ques_embed * pre_mult_img_feat
        
		out = torch.sigmoid(self.fcs(final_feats)).squeeze()

		return out


def train_model(train_dataloader, optimizer, loss, net, validation_loader, is_distributed, use_cuda, train_img_dict, train_obj_dict, val_img_dict, val_obj_dict, log='runs', epochs=5):

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
		# predictions = []
		# actual = []
		for i,batch in enumerate(train_dataloader):
			if i % 100 == 0 and i != 0:
				print('\nAccuracy: '+str(float(correct)/total)+' Loss: '+str(running_loss/total))
			q, q_len = batch.Question
			q = q.to(device)
			y = batch.Answer.to(device)
			imgids = batch.ImageID
			img_vec = torch.zeros(0, dtype = torch.float)
			obj_vec = torch.zeros(0, dtype = torch.float)
			for imgid in imgids:
				img = train_img_dict[int(imgid)].unsqueeze(0)
				obj = train_obj_dict[int(imgid)].unsqueeze(0).float()
				img_vec = torch.cat((img_vec, img), dim = 0)
				obj_vec = torch.cat((obj_vec, obj), dim = 0)
			img_vec = img_vec.to(device)
			obj_vec = obj_vec.to(device)
			 #Get the object and image features using the id      
 
			# zero the parameter gradients
			optimizer.zero_grad()

			y_pred = net(q, q_len, img_vec, obj_vec)

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
		print("\nRunning Validation")     
		val_correct = 0
		val_total = 0
		val_loss = 0
		with torch.set_grad_enabled(False):
			for batch in validation_loader:
				# Transfer to GPU
				q, q_len = batch.Question
				q = q.to(device)
				y = batch.Answer.to(device)
				imgids = batch.ImageID
				img_vec = torch.zeros(0, dtype = torch.float)
				obj_vec = torch.zeros(0, dtype = torch.float)
				for imgid in imgids:
					img = val_img_dict[int(imgid)].unsqueeze(0)
					obj = val_obj_dict[int(imgid)].unsqueeze(0).float()
					img_vec = torch.cat((img_vec, img), dim = 0)
					obj_vec = torch.cat((obj_vec, obj), dim = 0)
				img_vec = img_vec.to(device)
				obj_vec = obj_vec.to(device)
				# Model computations
				y_pred = net(q, q_len, img_vec, obj_vec)
				preds = (y_pred>0.5).float()
				val_correct += (torch.sum(preds==y)).item()
				val_total += len(y)
				loss_val = loss(y_pred, y)
				val_loss += loss_val.item()
		print('Validation accuracy: '+str(float(val_correct)/val_total))
		validation_losses[epoch+1] = val_loss/val_total
		validation_accuracies[epoch+1] = float(val_correct)/val_total
		torch.save(net.cpu().state_dict(), 'dl_advance_log/with_yolo/' + str(epoch) +'.pt')
		net = net.to(device)
        
	if not os.path.exists('dl_advance_log/with_yolo'):
		os.makedirs('dl_advance_log/with_yolo')
	np.save('dl_advance_log/with_yolo/train_loss.npy', training_losses, allow_pickle=True)
	np.save('dl_advance_log/with_yolo/train_acc.npy', training_accuracies, allow_pickle=True)
	np.save('dl_advance_log/with_yolo/val_loss.npy', validation_losses, allow_pickle=True)
	np.save('dl_advance_log/with_yolo/val_acc.npy', validation_accuracies, allow_pickle=True)



class FinalDataset(Dataset):
	def __init__(self, vqa_data):
		self.data = vqa_data
		return

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		q, a, image_id = self.data[idx]
		return q[0], torch.tensor([a], dtype=torch.float), image_id

class vqaDataset(Dataset):
	def __init__(self, vqa, annotations, dataSubType):
		self.imgIds = vqa.getImgIds()
		self.quesIds = vqa.getQuesIds()
		self.dataSubType = dataSubType
		self.annotations = annotations
		self.vqa = vqa
		self.image_vecs = {}
		self.obj_vecs = {}
		for i, imgId_ in enumerate(set(self.imgIds)):
			imgFilename = 'COCO_' + self.dataSubType + '_' + str(imgId_).zfill(12) + '.pt'
			imgFilename_rcnn = 'COCO_' + str(imgId_).zfill(12) + '.pt'
			image_vec = torch.load('fasterrcnn_output/'+ imgFilename_rcnn, map_location=torch.device('cpu')).squeeze(0).squeeze(1).squeeze(1)
			obj_vec = torch.load('yolo_feats/'+ self.dataSubType + '/'+ self.dataSubType + '/' + imgFilename, map_location=torch.device('cpu')) 
			self.image_vecs[imgId_] = image_vec
			self.obj_vecs[imgId_] = obj_vec
			if i % 10000 == 0:
				print(i)
		return
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
		return questions, answer, torch.tensor(imgId)
    
def create_train_loader(feat_dir, is_distributed, **kwargs):

	versionType ='v2_' 
	taskType    ='OpenEnded_' 
	dataType    ='mscoco_'
	dataSubType ='train2014_'

	annFile     ='%s%s%sannotations.json'%(versionType, dataType, dataSubType)
	quesFile    ='%s%s%s%squestions.json'%(versionType, taskType, dataType, dataSubType)

	vqa=VQA(annFile, quesFile)

	annotations = np.load('training_annotations.npy', allow_pickle = True).tolist()

	vqa_data = vqaDataset(vqa, annotations, "train2014")
	image_vecs = vqa_data.image_vecs 
	obj_vecs = vqa_data.obj_vecs
	# Insert location of image features
	dataset = FinalDataset(vqa_data)

	loader = DataLoader(dataset, batch_size=1)

	return loader, image_vecs, obj_vecs

def create_val_loader(feat_dir, **kwargs):

	versionType ='v2_' 
	taskType    ='OpenEnded_' 
	dataType    ='mscoco_'
	dataSubType ='val2014_'

	annFile     ='%s%s%sannotations.json'%(versionType, dataType, dataSubType)
	quesFile    ='%s%s%s%squestions.json'%(versionType, taskType, dataType, dataSubType)

	vqa=VQA(annFile, quesFile)
	annotations = filter_yes_no(vqa, dataSubType, bad_ids=np.load('bad_im_files_val.npy').tolist())

	vqa_data = vqaDataset(vqa, annotations, "val2014")
	image_vecs = vqa_data.image_vecs 
	obj_vecs = vqa_data.obj_vecs
	# Insert location of image features
	dataset = FinalDataset(vqa_data)

	loader = DataLoader(dataset, batch_size=1, shuffle=True)
	return loader, image_vecs, obj_vecs

def get_dataset(loader, fields):
	example_list = []
	print(len(loader))
	for i, (q, a, img_id) in enumerate(loader):
		if i % 100 == 0:
			print(i)
		for q_i, a_i, img_id_i in zip(q, a, img_id):
			example_list.append(torchtext.data.Example.fromlist([q_i, a_i, img_id_i.detach().numpy()], fields))

	complete_set = torchtext.data.Dataset(example_list, fields)
	return complete_set

def combine_datasets(train_loader, val_loader):
	TEXT = torchtext.data.Field(sequential=True, lower=True, include_lengths=True)
	ANS = torchtext.data.Field(sequential=False, use_vocab=False, dtype=torch.float, is_target=True)
	IMGID = torchtext.data.Field(sequential=False, use_vocab=False, dtype=torch.float)

	fields = [('Question', TEXT),  ('Answer', ANS), ('ImageID', IMGID)]

	train_set = get_dataset(train_loader, fields)
	print("train_set made\n")   
	val_set = get_dataset(val_loader, fields)
	print("val set made\n")
	TEXT.build_vocab(train_set, val_set, vectors='glove.6B.300d')
	print("vocab_built")
	train_iter = torchtext.data.BucketIterator(
		train_set, 
		batch_size = 100,
		sort_key = lambda x: len(x.Question),
		sort_within_batch = True,
		repeat=False, 
		shuffle=True)
	print("train_iter built")
	# Define the test iterator
	val_iter = torchtext.data.BucketIterator(
		val_set, 
		batch_size = 100,
		sort=False,
		sort_key = lambda x: len(x.Question),
		sort_within_batch = False,
		repeat=False, 
		shuffle=False)
	print("val_iter built")
	return train_iter, val_iter, len(TEXT.vocab), TEXT.vocab.vectors


def printTime(text):
	now = datetime.now()
	current_time = now.strftime("%H:%M:%S")
	print(current_time, " : " + text, )    
    
if __name__ == '__main__':
	printTime("Training Start.........")

	data_dir = 'test'
	seed = 12423352351
	is_distributed= False
    
	use_cuda=torch.cuda.is_available()

	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
	# set the seed for generating random numbers
	torch.manual_seed(seed)
	if use_cuda:
		torch.cuda.manual_seed(seed)

	train_loader, train_img_dict, train_obj_dict = create_train_loader(data_dir, is_distributed, **kwargs)
	val_loader, val_img_dict, val_obj_dict = create_val_loader(data_dir, **kwargs)

	train_iter, val_iter, vocab_size, word_embeddings = combine_datasets(train_loader, val_loader)

	printTime("Data set created.........")
    
	hidden_size = 1280
	embedding_length = 300
	img_size = 2048

    
	dl_net = Network(img_size, hidden_size, vocab_size, embedding_length, word_embeddings)
	printTime("Network created.........")
    
	#### Modify everything below this with the complete network model ####

	criterion_log = nn.BCELoss(size_average=True)
    
	parameters = filter(lambda p: p.requires_grad, dl_net.parameters())
	optimizer_log = optim.Adamax(parameters)
	epochs = 20
	train_model(train_iter, optimizer_log, criterion_log, dl_net, val_iter, is_distributed, use_cuda,train_img_dict, train_obj_dict, val_img_dict, val_obj_dict, 'dl_advance_with_yolo', epochs=epochs)
    
	printTime("Finished training.........")
    
	torch.save(dl_net.cpu().state_dict(), 'dl_advance_log/with_yolo/' + str(epochs) +'.pt')
    
	printTime("Trained Network saved.........")


