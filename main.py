import time
import os
import numpy as np
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import warnings
from model import RelationalGraphModel
from data_utils import *
from evaluate import *
# from pytorchtools import EarlyStopping
import argparse
warnings.filterwarnings("ignore")
from collections import namedtuple
Formatted_pred = namedtuple('Formatted_pred', 'pred buggy_code_paths')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}\n')

class Train:
    def __init__(self, args):
        self.args = args
        # Load data A:A[0] is the R-C matrix, A[1] is the R-R matrix, A[2] is the C-C matrix; 
		# train_data contains only positive samples
        self.A, self.train_data, self.val_data, self.test_data, self.report_num, self.code_num, self.train_mat, \
        self.val_buggy_files, self.test_buggy_files = load_data(dirname='./data/' + self.args.data + '/')
        # load init embeddings: Report matrix and Code file matrix
        self.R, self.C = read_feature(dirname='./data/' + self.args.data + '/')

        # construct the train and test datasets
        self.train_dataset = BPRData(
            self.train_data, self.code_num, self.train_mat, self.args.num_ng, True)
        self.train_loader = data.DataLoader(self.train_dataset,
                                       batch_size=self.args.batch_size, shuffle=True, num_workers=4)  
        self.val_dataset = BPRData(
            self.val_data, self.code_num, self.train_mat, 0, False)  
        self.val_loader = data.DataLoader(self.val_dataset,
                                           batch_size=self.code_num, shuffle=False, num_workers=0)
        self.test_dataset = BPRData(
            self.test_data, self.code_num, self.train_mat, 0, False)  
        self.test_loader = data.DataLoader(self.test_dataset,
                                           batch_size=self.code_num, shuffle=False, num_workers=0)

        # Create Model
        self.model = RelationalGraphModel(
            report_size=self.report_num,
            code_size=self.code_num,
            report_features=self.R,
            code_features=self.C,
            feat_dim=self.R.shape[1],
            cuda=self.args.using_cuda,
        )
        print(
            "Loaded %s dataset with %d reports, %d code files, train data size is: %d, test data size is: %d"
            % (self.args.data, self.report_num, self.code_num, len(self.train_data) * self.args.num_ng, len(self.test_data) / self.code_num)
        )

        # Loss and optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.args.lr, weight_decay=self.args.lamda)

        # initialize the early_stopping object
        if self.args.using_cuda:
            print("Using the GPU")
            self.model.cuda()

    def train(self, epoch):
        # Start training
        self.model.train()
        self.train_loader.dataset.ng_sample()
        # Load a batch of data
        count = 0
        for report, code_i, code_j in self.train_loader:
            report = report.cuda()
            code_i = code_i.cuda()
            code_j = code_j.cuda()

            start_batch_time = time.time()
            self.model.zero_grad()
            prediction_i, prediction_j = self.model(self.A, report, code_i, code_j)
            loss = - (prediction_i - prediction_j).sigmoid().log().sum()
            loss.backward()
            self.optimizer.step()
            count += 1
            print(
                "Epoch: {:d}".format(epoch),
                "batch: {:d}".format(count),
                "train loss: {:.4f}".format(loss.item()),
                "batch cost: {:.4f}s".format(time.time() - start_batch_time),
            )

    def test(self, data, dataset, buggy_files, tag):
        with torch.no_grad():
            if tag == 'test':
                self.model.load_state_dict(torch.load(self.args.model_path))  # load model
            t = time.time()
            self.model.eval()
            all_format_pred = []
            test_count = 0
            for report, code_i, code_j in dataset:
                t0 = time.time()
                report = report.cuda()
                code_i = code_i.cuda()
                code_j = code_j.cuda()  # not useful when testing

                prediction_i, prediction_j = self.model(self.A, report, code_i, code_j)

                all_format_pred.append(
                    Formatted_pred(pred=prediction_i, buggy_code_paths=buggy_files[test_count])
                )
                test_count += 1
                print("Test total batch: {:.1f}".format(len(data) / self.code_num),
                      "test batch: {:d}".format(test_count),
                      "test cost is {:.4f}s".format(time.time() - t0))
            evaluator = Evaluator()
            ranked_predict = evaluator.rank(all_format_pred)
            hit_k, mean_ap, mean_rr = evaluator.evaluate(ranked_predict)

            elapsed_time = time.time() - t
            print("test cost is {:.4f}s".format(elapsed_time))
            print(f'MAP:   {mean_ap:.4f}')
            print(f'MRR:   {mean_rr:.4f}')
            for n, hit in enumerate(hit_k):
                print(f'hit_{n + 1}: {hit:.4f}')

            return mean_ap, mean_rr, hit_k

def parse_args():
    parser = argparse.ArgumentParser(description="Run RGCN.")
    parser.add_argument('--data', type=str, choices=['aspectj', 'eclipseUI', 'jdt', 'swt', 'tomcat'], default='aspectj', help="dataset.")
    parser.add_argument('--epochs', type=int, default=100, help="training epoches")
    parser.add_argument('--batch_size', type=int, default=512, help="batch size for training")
    parser.add_argument('--top_k', type=int, default=10, help="compute metrics@top_k")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--lamda', type=float, default=0.001, help="model regularization rate")
    parser.add_argument('--num_ng', type=int, default=4, help="sample negative items for training")
    parser.add_argument('--early_stop', type=int, default=1, help='Whether to stop training early.')
    parser.add_argument("--out", default=True, help="save model or not")
    parser.add_argument("--model_path", default='./data/jdt/model/RC.pth', help="save model or not")
    parser.add_argument('--drop', type=float, default=0.1)
    parser.add_argument("--no_cuda", action="store_true", default=False, help="Enables CUDA training.")
    parser.add_argument("--gpu", type=str, default="0", help="gpu card ID")
	parser.add_argument("--just_test", default=True, help="just test the model")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.using_cuda = not args.no_cuda and torch.cuda.is_available()
    if args.using_cuda:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
    return args

if __name__ == "__main__":
    strat = time.time()
    args = parse_args()
    train = Train(args)
	
	if args.just_test:
        print(f'Evaluation Result:')
        train.test(train.test_data, train.test_loader, train.test_buggy_files, 'test')
		
	else:
		best_map, best_mrr, best_hit, best_epoch = 0, 0, 0, 0
		for epoch in range(args.epochs):
			# train
			train.train(epoch)
			if (epoch + 1) > 40:
				mean_ap, mean_rr, hit_k = train.test(train.val_data, train.val_loader, train.val_buggy_files, 'val')
				if mean_ap > best_map:
					best_map, best_mrr, best_hit, best_epoch = mean_ap, mean_rr, hit_k, epoch
					if args.out:
						if not os.path.exists('./data/' + args.data + '/model/'):
							os.mkdir('./data/' + args.data + '/model/')
						torch.save(train.model.state_dict(), args.model_path)

		print(f'Best epoch: {best_epoch}\n')
		print(f'Best MAP:   {best_map:.4f}')
		print(f'Best MRR:   {best_mrr:.4f}')
		for n, hit in enumerate(best_hit):
			print(f'Best hit_{n + 1}: {hit:.4f}')

		print(f'Evaluation Result:')
		train.test(train.test_data, train.test_loader, train.test_buggy_files, 'test')

		print('total cost (train cost + test cost): {:.4f}s'.format(time.time() - strat))

