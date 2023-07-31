import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random

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
		# Find the neighbors of the target node
        mask_1_code, mask_2_code_report, mask_2_code_code, \
        unique_code_neighs_1, unique_code_report_neighs_2, unique_code_code_neighs_2, \
        mask_1_report, mask_2_report_code, mask_2_report_report, \
        unique_report_neighs_1, unique_report_code_neighs_2, unique_report_report_neighs_2 \
            = self.get_neighs_batch(A, batch_nodes, mode)

        # Aggregate second-order neighbors and update first-order neighbors
        new_code_emb = self.aggregate_code_two_order(mask_1_code, mask_2_code_report, mask_2_code_code,
                                                     unique_code_neighs_1, unique_code_report_neighs_2,
                                                     unique_code_code_neighs_2)
        new_report_emb = self.aggregate_report_two_order(mask_1_report, mask_2_report_report, mask_2_report_code,
                                                         unique_report_neighs_1, unique_report_code_neighs_2,
                                                         unique_report_report_neighs_2)
        # Aggregate first-order neighbors and update the target node
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

        # second-order neighbors
        # code-report [len(unique_code_neighs_1), total number of reports]
        code_report_neighs_2 = report_code_A.T[unique_code_neighs_1]
        row_report_2, col_report_2 = np.nonzero(code_report_neighs_2)
        unique_code_report_neighs_2 = list(set(col_report_2.tolist()))
        # [len(unique_code_neighs_1), len(unique_code_report_neighs_2)]
        mask_2_code_report = code_report_neighs_2[:, unique_code_report_neighs_2]

        # code-code [len(unique_code_neighs_1), total number of code files]
        code_code_neighs_2 = code_code_A[unique_code_neighs_1]
        row_code_2, col_code_2 = np.nonzero(code_code_neighs_2)
        unique_code_code_neighs_2 = list(set(col_code_2.tolist()))
        # [len(unique_code_neighs_1), len(unique_code_report_neighs_2)]
        mask_2_code_code = code_code_neighs_2[:, unique_code_code_neighs_2]

        # report_neighs_1：
        row_report_1, col_report_1 = np.nonzero(report_neighs_1)
        unique_report_neighs_1 = list(set(col_report_1.tolist()))
        # [512, len(unique_report_neighs_1)]
        mask_1_report = report_neighs_1[:, unique_report_neighs_1]

        # second-order neighbors
        # report-code [len(unique_report_neighs_1), total number of code files]
        report_code_neighs_2 = report_code_A[unique_report_neighs_1]
        row_code_2, col_code_2 = np.nonzero(report_code_neighs_2)
        unique_report_code_neighs_2 = list(set(col_code_2.tolist()))
        # [len(unique_code_neighs_1), len(unique_code_report_neighs_2)]
        mask_2_report_code = report_code_neighs_2[:, unique_report_code_neighs_2]

        # report-report [len(unique_report_neighs_1), total number of reports]
        report_report_neighs_2 = report_report_A[unique_report_neighs_1]
        row_report_2, col_report_2 = np.nonzero(report_report_neighs_2)
        unique_report_report_neighs_2 = list(set(col_report_2.tolist()))
        # [len(unique_code_neighs_1), len(unique_code_report_neighs_2)]
        mask_2_report_report = report_report_neighs_2[:, unique_report_report_neighs_2]

        return mask_1_code, mask_2_code_report, mask_2_code_code, \
               unique_code_neighs_1, unique_code_report_neighs_2, unique_code_code_neighs_2, \
               mask_1_report, mask_2_report_code, mask_2_report_report, \
               unique_report_neighs_1, unique_report_code_neighs_2, unique_report_report_neighs_2

    def aggregate_code_two_order(self, np_mask_1_code, np_mask_2_report, np_mask_2_code, unique_code_neighs_1, unique_code_report_neighs_2, unique_code_code_neighs_2):
        mask_1_code = Variable(torch.zeros(np_mask_1_code.shape[0], len(unique_code_neighs_1)))
        mask_2_report = Variable(torch.zeros(len(unique_code_neighs_1), len(unique_code_report_neighs_2)))
        mask_2_code = Variable(torch.zeros(len(unique_code_neighs_1), len(unique_code_code_neighs_2)))
        
        row_indices_1_code, column_indices_1_code = np.nonzero(np_mask_1_code)
        row_indices_2_report, column_indices_2_report = np.nonzero(np_mask_2_report)
        row_indices_2_code, column_indices_2_code = np.nonzero(np_mask_2_code)

        
        mask_1_code[row_indices_1_code, column_indices_1_code] = 1
        mask_2_report[row_indices_2_report, column_indices_2_report] = 1
        mask_2_code[row_indices_2_code, column_indices_2_code] = 1
        if self.cuda:
            mask_1_code = mask_1_code.cuda()
            mask_2_report = mask_2_report.cuda()
            mask_2_code = mask_2_code.cuda()

        
        num_neigh_1_code = mask_1_code.sum(1, keepdim=True)
        num_neigh_2_report = mask_2_report.sum(1, keepdim=True)
        num_neigh_2_code = mask_2_code.sum(1, keepdim=True)
        
        mask_1_code = mask_1_code.div(num_neigh_1_code)
        mask_2_report = mask_2_report.div(num_neigh_2_report)
        mask_2_code = mask_2_code.div(num_neigh_2_code)
        mask_1_code[mask_1_code != mask_1_code] = 0
        mask_2_report[mask_2_report != mask_2_report] = 0
        mask_2_code[mask_2_code != mask_2_code] = 0

        if self.cuda:
            # aggregate second-order neighbors information
            embed_matrix_2_report = self.R(torch.LongTensor(unique_code_report_neighs_2).cuda())
            embed_matrix_2_code = self.C(torch.LongTensor(unique_code_code_neighs_2).cuda())
        else:
            # aggregate second-order neighbors information
            embed_matrix_2_report = self.R(torch.LongTensor(unique_code_report_neighs_2))
            embed_matrix_2_code = self.C(torch.LongTensor(unique_code_code_neighs_2))
        
        embed_matrix_1_code_report = mask_2_report.mm(embed_matrix_2_report)  
        embed_matrix_1_code_code = mask_2_code.mm(embed_matrix_2_code)  
        # aggregate first-order neighbors information
        code_report_emb = mask_1_code.mm(embed_matrix_1_code_report)  
        code_code_emb = mask_1_code.mm(embed_matrix_1_code_code)  
        new_code_emb = code_report_emb * self.gamma_code + code_code_emb * (1 - self.gamma_code)

        return new_code_emb

    def aggregate_report_two_order(self, np_mask_1_report, np_mask_2_report, np_mask_2_code, unique_report_neighs_1, unique_report_code_neighs_2, unique_report_report_neighs_2):
        mask_1_report = Variable(torch.zeros(np_mask_1_report.shape[0], len(unique_report_neighs_1)))
        mask_2_code = Variable(torch.zeros(len(unique_report_neighs_1), len(unique_report_code_neighs_2)))
        mask_2_report = Variable(torch.zeros(len(unique_report_neighs_1), len(unique_report_report_neighs_2)))
        
        row_indices_1_report, column_indices_1_report = np.nonzero(np_mask_1_report)
        row_indices_2_code, column_indices_2_code = np.nonzero(np_mask_2_code)
        row_indices_2_report, column_indices_2_report = np.nonzero(np_mask_2_report)

        mask_1_report[row_indices_1_report, column_indices_1_report] = 1
        mask_2_code[row_indices_2_code, column_indices_2_code] = 1
        mask_2_report[row_indices_2_report, column_indices_2_report] = 1
        if self.cuda:
            mask_1_report = mask_1_report.cuda()
            mask_2_code = mask_2_code.cuda()
            mask_2_report = mask_2_report.cuda()

        num_neigh_1_report = mask_1_report.sum(1, keepdim=True)
        num_neigh_2_code = mask_2_code.sum(1, keepdim=True)
        num_neigh_2_report = mask_2_report.sum(1, keepdim=True)
       
        mask_1_report = mask_1_report.div(num_neigh_1_report)
        mask_2_code = mask_2_code.div(num_neigh_2_code)
        mask_2_report = mask_2_report.div(num_neigh_2_report)
        mask_1_report[mask_1_report != mask_1_report] = 0
        mask_2_code[mask_2_code != mask_2_code] = 0
        mask_2_report[mask_2_report != mask_2_report] = 0

        if self.cuda:
            # aggregate second-order neighbors information
            embed_matrix_2_code = self.C(torch.LongTensor(unique_report_code_neighs_2).cuda())
            embed_matrix_2_report = self.R(torch.LongTensor(unique_report_report_neighs_2).cuda())
        else:
            # aggregate second-order neighbors information
            embed_matrix_2_code = self.C(torch.LongTensor(unique_report_code_neighs_2))
            embed_matrix_2_report = self.R(torch.LongTensor(unique_report_report_neighs_2))
        
        embed_matrix_1_report_code = mask_2_code.mm(embed_matrix_2_code)  
        embed_matrix_1_report_report = mask_2_report.mm(embed_matrix_2_report)  
        # aggregate first-order neighbors information
        report_code_emb = mask_1_report.mm(embed_matrix_1_report_code) 
        report_report_emb = mask_1_report.mm(embed_matrix_1_report_report)  
        new_report_emb = report_code_emb * self.gamma_report + report_report_emb * (1 - self.gamma_report)

        return new_report_emb

