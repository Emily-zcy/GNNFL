import numpy as np
import torch
from collections import OrderedDict

def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, A, test_loader, top_k):
	HR, NDCG = [], []

	for report, code_i, code_j in test_loader:
		report = report.cuda()
		code_i = code_i.cuda()
		code_j = code_j.cuda() # not useful when testing

		prediction_i, prediction_j = model(A, report, code_i, code_j)
		_, indices = torch.topk(prediction_i, top_k)
		recommends = torch.take(
				code_i, indices).cpu().numpy().tolist()

		gt_item = code_j[0].item()
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))

	return np.mean(HR), np.mean(NDCG)

class Evaluator(object):
    def __init__(self):
        self.buggy_code_paths = []

    def rank(self, formatted_predict):
        pred_results = [each.pred for each in formatted_predict]
        self.buggy_code_paths = [each.buggy_code_paths for each in formatted_predict]

        ranked_result = []
        for each_report_pred_result in pred_results:
            # code id: predict value
            each_report_pred_result_dict = dict(enumerate(each_report_pred_result.cpu().numpy().tolist()))
            # sort by value 
            each_report_pred_result_dict = dict(sorted(each_report_pred_result_dict.items(), key=lambda x: x[1], reverse=True))
            each_ranked_result = {}
            for index, code_id in enumerate(each_report_pred_result_dict.keys()):
                each_ranked_result[code_id] = index + 1
            ranked_result.append(each_ranked_result)
        return ranked_result

    def evaluate(self, ranked_result):
        hit_k = self.cal_hit_k(ranked_result)
        mean_ap = self.cal_map(ranked_result)
        mean_rr = self.cal_mrr(ranked_result)
        return hit_k, mean_ap, mean_rr

    def cal_hit_k(self, ranked_result, K=10):
        at_k = [0] * K
        num_report = len(ranked_result)
        for i, rank_info in enumerate(ranked_result):
            buggy_code_paths = self.buggy_code_paths[i]
            top_rank = min([rank_info[path] for path in buggy_code_paths])
            if top_rank <= K:
                at_k[top_rank - 1] += 1

        hit_k = [sum(at_k[:i + 1]) / num_report for i in range(K)]
        return hit_k

    def cal_map(self, ranked_result):
        """Mean Average Precision"""
        avg_p = []
        for i, rank_info in enumerate(ranked_result):
            buggy_code_paths = self.buggy_code_paths[i]
            buggy_code_ranks = list(sorted([rank_info[path] for path in buggy_code_paths]))
            precision_k = [(i + 1) / rank for i, rank in enumerate(buggy_code_ranks)]
            avg_p.append(sum(precision_k) / len(buggy_code_ranks))
        mean_avg_p = sum(avg_p) / len(ranked_result)
        return mean_avg_p

    def cal_mrr(self, ranked_result):
        """Mean Reciprocal Rank"""
        reciprocal_rank = []
        for i, rank_info in enumerate(ranked_result):
            buggy_code_paths = self.buggy_code_paths[i]
            top_rank = min([rank_info[path] for path in buggy_code_paths])
            reciprocal_rank.append(1 / top_rank)
        mrr = sum(reciprocal_rank) / len(ranked_result)
        return mrr
