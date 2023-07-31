import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class RelationalGraphLayer(nn.Module):
    def __init__(
        self,
            report_size,
            code_size,
            report_features,
            code_features,
            feat_dim,
            cuda=False
    ):
        super(RelationalGraphLayer, self).__init__()
        self.report_size = report_size
        self.code_size = code_size
        self.R = report_features
        self.C = code_features
        self.feat_dim = feat_dim
        self.cuda = cuda

        self.gamma_code = nn.Parameter(torch.FloatTensor(size=(1,)), requires_grad=True)
        self.gamma_report = nn.Parameter(torch.FloatTensor(size=(1,)), requires_grad=True)
        self.gamma = nn.Parameter(torch.FloatTensor(size=(1,)), requires_grad=True)
        nn.init.ones_(self.gamma_code)
        nn.init.ones_(self.gamma_report)
        nn.init.ones_(self.gamma)  

    def forward(self, A, batch_nodes, mode):
        mask_1_code, unique_code_neighs_1, mask_1_report, unique_report_neighs_1 \
            = self.get_neighs_batch(A, batch_nodes, mode)

        new_code_emb = self.aggregate_code_one_order(mask_1_code, unique_code_neighs_1)
        new_report_emb = self.aggregate_report_one_order(mask_1_report, unique_report_neighs_1)
        new_emb = new_code_emb * self.gamma + new_report_emb * (1 - self.gamma)

        return new_emb

    def get_neighs_batch(self, A, batch_index, mode):
        report_code_A = A[0].toarray()
        report_report_A = A[1].toarray()  
        code_code_A = A[2].toarray()  
        if mode == "report":
            # first-order neighbors
            code_neighs_1 = report_code_A[batch_index.cpu()]  
            report_neighs_1 = report_report_A[batch_index.cpu()]  
        elif mode == "code":
            report_neighs_1 = report_code_A.T[batch_index.cpu()]  
            code_neighs_1 = code_code_A[batch_index.cpu()]  

        # code_neighs_1：
        row_code_1, col_code_1 = np.nonzero(code_neighs_1)
        unique_code_neighs_1 = list(set(col_code_1.tolist()))
        # [512, len(unique_code_neighs_1)]
        mask_1_code = code_neighs_1[:, unique_code_neighs_1]

        # report_neighs_1：
        row_report_1, col_report_1 = np.nonzero(report_neighs_1)
        unique_report_neighs_1 = list(set(col_report_1.tolist()))
        # [512, len(unique_report_neighs_1)]
        mask_1_report = report_neighs_1[:, unique_report_neighs_1]

        return mask_1_code, unique_code_neighs_1, mask_1_report, unique_report_neighs_1

    def aggregate_code_one_order(self, np_mask_1_code, unique_code_neighs_1):
        mask_1_code = Variable(torch.zeros(np_mask_1_code.shape[0], len(unique_code_neighs_1)))
        row_indices_1_code, column_indices_1_code = np.nonzero(np_mask_1_code)

        mask_1_code[row_indices_1_code, column_indices_1_code] = 1
        if self.cuda:
            mask_1_code = mask_1_code.cuda()

        num_neigh_1_code = mask_1_code.sum(1, keepdim=True)
        
        mask_1_code = mask_1_code.div(num_neigh_1_code)
        mask_1_code[mask_1_code != mask_1_code] = 0

        if self.cuda:
            embed_matrix_1_code = self.C(torch.LongTensor(unique_code_neighs_1).cuda())
        else:
            embed_matrix_1_code = self.C(torch.LongTensor(unique_code_neighs_1))
        code_emb = mask_1_code.mm(embed_matrix_1_code)  
        return code_emb

    def aggregate_report_one_order(self, np_mask_1_report, unique_report_neighs_1):
        mask_1_report = Variable(torch.zeros(np_mask_1_report.shape[0], len(unique_report_neighs_1)))
        row_indices_1_report, column_indices_1_report = np.nonzero(np_mask_1_report)

        mask_1_report[row_indices_1_report, column_indices_1_report] = 1
        if self.cuda:
            mask_1_report = mask_1_report.cuda()

        num_neigh_1_report = mask_1_report.sum(1, keepdim=True)
        
        mask_1_report = mask_1_report.div(num_neigh_1_report)
        mask_1_report[mask_1_report != mask_1_report] = 0

        if self.cuda:
            embed_matrix_1_report = self.R(torch.LongTensor(unique_report_neighs_1).cuda())
        else:
            embed_matrix_1_report = self.R(torch.LongTensor(unique_report_neighs_1))
        report_emb = mask_1_report.mm(embed_matrix_1_report)  
        return report_emb


