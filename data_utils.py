import numpy as np 
import pandas as pd
import pickle
import scipy.sparse as sp
from itertools import cycle
from tqdm import tqdm

import torch
import torch.utils.data as data

from collections import defaultdict


def load_data(dirname):
	A = []
	with open(dirname + "adj/report_code_csr_A", 'rb') as r_c_adj:
		report_code_csr_A = pickle.load(r_c_adj)  
	A.append(report_code_csr_A)

	with open(dirname + "adj/report_report_csr_A", 'rb') as r_r_adj:
		report_report_csr_A = pickle.load(r_r_adj)  
	A.append(report_report_csr_A)

	with open(dirname + "adj/code_code_csr_A", 'rb') as c_c_adj:
		code_code_csr_A = pickle.load(c_c_adj)   
	A.append(code_code_csr_A)

	with open(dirname + 'dataset/dataset_for_train.pkl', 'rb') as fr_train:
		train_data = pickle.load(fr_train)  
	with open(dirname + 'dataset/dataset_for_val.pkl', 'rb') as fr_test:
		val_data = pickle.load(fr_test)  
	with open(dirname + 'dataset/buggyfiles_for_val.pkl', 'rb') as fr_buggy_files:
		val_buggy_files = pickle.load(fr_buggy_files)  
	with open(dirname + 'dataset/dataset_for_test.pkl', 'rb') as fr_test:
		test_data = pickle.load(fr_test)  
	with open(dirname + 'dataset/buggyfiles_for_test.pkl', 'rb') as fr_buggy_files:
		test_buggy_files = pickle.load(fr_buggy_files)  

	report_code_train = defaultdict(set)
	for train_sample in train_data:
		report_code_train[train_sample[0]].add(train_sample[1])

	return A, train_data, val_data, test_data, A[0].shape[0], A[0].shape[1], report_code_train, val_buggy_files, test_buggy_files

def read_feature(dirname):
	with open(dirname + 'report_corpus/report_features.pkl','rb') as fr_report:
		report_feature_list = pickle.load(fr_report)
	with open(dirname + 'code_corpus/code_features.pkl', 'rb') as fr_code:
		code_feature_list = pickle.load(fr_code)

	report_feature_list = torch.tensor(report_feature_list, dtype=torch.float)
	code_feature_list = torch.tensor(code_feature_list, dtype=torch.float)
	print('report_feature:', report_feature_list.shape)
	print('code_feature:', code_feature_list.shape)

	return report_feature_list, code_feature_list

class BPRData(data.Dataset):
	def __init__(self, features, 
				num_code, train_mat=None, num_ng=0, is_training=None):
		super(BPRData, self).__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		self.features = features
		self.num_code = num_code
		self.train_mat = train_mat
		self.num_ng = num_ng
		self.is_training = is_training

	def ng_sample(self):
		assert self.is_training, 'no need to sampling when testing'

		self.features_fill = []
		for x in self.features:
			r, i = x[0], x[1]
			for t in range(self.num_ng):
				j = np.random.randint(self.num_code)
				while j in self.train_mat[r]:
					j = np.random.randint(self.num_code)
				self.features_fill.append([r, i, j])

	def __len__(self):
		return self.num_ng * len(self.features) if self.is_training else len(self.features)

	def __getitem__(self, idx):
		features = self.features_fill if self.is_training else self.features

		report = features[idx][0]
		code_i = features[idx][1]
		# code_i and code_j is the same in test set
		code_j = features[idx][2] if self.is_training else features[idx][1]  

		return report, code_i, code_j

