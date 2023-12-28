import torch.nn as nn
import torch
import time
import pandas as pd
import os
import numpy as np
import torch.nn.functional as F
import re
from tqdm import tqdm
from termcolor import colored, cprint

from .utils import *
from .model_utils import *
from .abstract_data import AbstractData

class AbstractRS(nn.Module):
    def __init__(self, args) -> None:
        super(AbstractRS, self).__init__()

        self.args = args
        self.parse_args(args) # parse the args
        self.preperation() # neccessary configuration (e.g., file directory)
        self.load_data() # load the data
        self.add_additional_args() # add additional args (e.g., data information)
        self.load_model() # load the model
        self.get_loss_function() # get the loss function (optional)
        self.set_optimizer() # set the optimizer

    def parse_args(self, args):
        self.args = args
        # General Args
        self.rs_type = args.rs_type
        self.model_name = args.model_name
        self.dataset_name = args.dataset

        self.device = torch.device(args.cuda)
        self.test_only = args.test_only
        self.clear_checkpoints = args.clear_checkpoints
        self.saveID = args.saveID

        self.seed = args.seed
        self.max_epoch = args.max_epoch
        self.verbose = args.verbose
        self.patience = args.patience

        # Model Args
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.hidden_size = args.hidden_size
        self.weight_decay = args.weight_decay
        self.dropout = args.dropout

    def preperation(self):
        self.base_path = './weights/{}/{}/{}/{}'.format(self.rs_type, self.dataset_name, self.model_name, self.saveID)
        ensureDir(self.base_path)
        self.start_epoch = 0
        self.best_valid_epoch = 0

    def load_data(self):
        self.data = AbstractData(self.args)

    def add_additional_args(self):
        # raise NotImplementedError
        return

    def load_model(self):
        exec('from models.LLM.'+ self.model_name + ' import ' + self.model_name)
        print('Model %s loaded!' % (self.model_name))
        self.model = eval(self.model_name + '(self.args)')
        self.model.to(self.device)
    
    def get_loss_function(self):
        return

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-8, weight_decay=self.weight_decay)
        # raise NotImplementedError

    # main entrance
    def execute(self):
        # write args
        perf_str = str(self.args)
        print(self.base_path + '_stats.txt')
        with open(self.base_path + '_stats.txt','a') as f:
            f.write(perf_str+"\n")

        # restore the checkpoint
        self.restore_checkpoint(self.model, self.base_path, self.device) 

        start_time = time.time()
        # train the model if not test only
        if not self.test_only:
            print("start training")
            self.train()
            # test the model
            print("start testing")
            self.restore_checkpoint(self.model, self.base_path, self.device) 
        end_time = time.time()

        print_str = "The best epoch is % d, total training cost is %.1f" % (max(self.best_valid_epoch, self.start_epoch), end_time - start_time)
        with open(self.base_path + '_stats.txt', 'a') as f:
            f.write(print_str + "\n")

        self.model.eval() # evaluate the best model
        self.final_evaluate()


    def train(self) -> None:
        # TODO
        self.total_step=0
        self.best_epoch = 0
        self.current_patience = 0
        self.stop_metric_max = 0

        self.stop_flag = False
        for epoch in range(self.start_epoch, self.max_epoch):
            if self.stop_flag: # early stop
                break

            t1=time.time()
            losses = self.train_one_epoch(epoch) # train one epoch
            t2=time.time()

            self.document_running_loss(losses, epoch, t2-t1) # report the loss with time
            if self.verbose >= 1 and (epoch + 1) % int(self.verbose) == 0: # evaluate the model
                self.eval_and_check_early_stop(epoch)

    def train_one_epoch(self, epoch):
        raise NotImplementedError
        # if()

    def document_running_loss(self, losses:list, epoch, t_one_epoch, prefix=""):
        loss_str = ', '.join(['%.5f']*len(losses)) % tuple(losses)
        perf_str = prefix + 'Epoch %d [%.1fs]: train==[' % (
                epoch, t_one_epoch) + loss_str + ']'
        with open(self.base_path + '_stats.txt','a') as f:
            f.write(perf_str+"\n")

    def eval_and_check_early_stop(self, epoch, epoch_progress=0):
        self.model.eval()
        cprint(f"[INFO] Start evaluating and checking early stop", color='green', attrs=['bold'])
        stop_metric = self.evaluate(self.model, self.data.test_data, self.device, name='valid')

        if stop_metric > self.stop_metric_max:
            self.stop_metric_max = stop_metric
            self.best_valid_epoch = epoch+epoch_progress
            self.current_patience = 0
            self.save_checkpoint(self.model, epoch, self.base_path, epoch_progress=epoch_progress)
        
        else:
            self.current_patience += 1
            if(self.current_patience >= self.patience):
                self.stop_flag = True
                print('Early stop at epoch %d' % epoch)
        
        print('Best checkpoint EPOCH:{}'.format(self.best_valid_epoch))
        self.model.train()
    
    def final_evaluate(self):
        self.model.eval()
        cprint(f"[INFO] Final evaluation", color='green', attrs=['bold'])
        self.evaluate(self.model, self.data.test_data, self.device, name='test')

    @torch.no_grad()
    def evaluate(self, model, eval_data, device, name='valid'):
        # TODO 改candidate形式
        # raise NotImplementedError
        cprint(f"[INFO] Start evaluating on {name} dataset", color='green', attrs=['bold'])
        cprint("[Warning] The func 'evaluation' is not implemented!", color='red', attrs=['bold'])
        return 0

    def save_checkpoint(self, model, epoch, checkpoint_dir, epoch_progress=0):
        # raise NotImplementedError
        cprint(f"[INFO] Start saving the checkpoint at epoch {epoch}", color='green', attrs=['bold'])
        cprint("[Warning] The func 'save_checkpoint' is not implemented!", color='red', attrs=['bold'])

    def restore_checkpoint(self, model, checkpoint_dir, device, force=False, pretrain=False):
        # raise NotImplementedError
        cprint("[INFO] Start restoring the checkpoint", color='green', attrs=['bold'])
        cprint("[Warning] The func 'restore_checkpoint' is not implemented!", color='red', attrs=['bold'])
