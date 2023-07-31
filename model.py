import torch
import torch.nn as nn
import torch.nn.functional as F
# aggregate R-C, R-R, C-C
# from aggregator import RelationalGraphLayer
# aggregate R-C
from aggregator_RC import RelationalGraphLayer
# aggregate R-C and R-R
# from aggregator_RC_RR import RelationalGraphLayer
# aggregate R-C and C-C
# from aggregator_RC_CC import RelationalGraphLayer
# aggregate one-hop neighbor information
# from aggregator_1_hop import RelationalGraphLayer


class RelationalGraphModel(nn.Module):
	def __init__(self,
				 report_size,
				 code_size,
				 report_features,
				 code_features,
				 feat_dim,
				 cuda=False):
		super(RelationalGraphModel, self).__init__()
		"""
		report_size: number of reports;
		code_size: number of code files.
		"""
		self.report_size = report_size
		self.code_size = code_size
		self.feat_dim = feat_dim

		self.R = nn.Embedding(report_size, feat_dim)
		self.C = nn.Embedding(code_size, feat_dim)
		# 初始化R和C矩阵
		self.R.weight = nn.Parameter(report_features)
		self.C.weight = nn.Parameter(code_features)

		self.relu = nn.ReLU()
		self.dropout = 0.1

		self.aggregator = RelationalGraphLayer(
			report_size,
			code_size,
			self.R,
			self.C,
			feat_dim,
			cuda=cuda)
		self.aggregator.cuda = cuda

	def forward(self, A, report_batch, code_i_batch, code_j_batch):
		R_r = self.aggregator.forward(A, report_batch, mode="report")  # report index
		C_i = self.aggregator.forward(A, code_i_batch, mode="code")  # code pos index
		C_j = self.aggregator.forward(A, code_j_batch, mode="code")  # code neg index

		R_r = F.dropout(self.relu(R_r), self.dropout, training=self.training)
		C_i = F.dropout(self.relu(C_i), self.dropout, training=self.training)
		C_j = F.dropout(self.relu(C_j), self.dropout, training=self.training)

		prediction_i = (R_r * C_i).sum(dim=-1)
		prediction_j = (R_r * C_j).sum(dim=-1)
		return prediction_i, prediction_j
