import torch.nn as nn
import torch
import time
import pandas as pd
import os
import numpy as np
import torch.nn.functional as F
import re
from tqdm import tqdm
from .utils import *
from .abstract_data import AbstractData

class AbstractRS(nn.Module):
    def __init__(self, args) -> None:
        super(AbstractRS, self).__init__()

        self.args = args
        self.parse_args(args)
        self.preperation()
        self.load_data()
        self.add_additional_args()
        self.load_model()
        self.get_loss_function()
        self.set_optimizer()
        

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

        # Sequential Model Args
        self.reward_click = args.r_click
        self.reward_buy = args.r_buy
        self.loss = args.loss

    def preperation(self):
        self.data_directory = './data/' + self.dataset_name
        self.saveID = self.saveID + "_batch_size=" + str(self.batch_size)\
                + "_lr=" + str(self.lr) 
        self.base_path = './weights/{}/{}/{}/{}'.format(self.rs_type, self.dataset_name, self.model_name, self.saveID)
        ensureDir(self.base_path)
        self.topk=[10, 20, 50]
        self.checkpoint_buffer=[]
        self.best_valid_epoch = 0

    def load_data(self):
        self.data = AbstractData(self.args)
        self.item_num = self.data.item_num
        self.seq_size = self.data.seq_size

    def add_additional_args(self):
        self.args.item_num = self.data.item_num
        self.args.state_size = self.data.seq_size
        print(self.args)

    def load_model(self):
        exec('from models.Seq.'+ self.model_name + ' import ' + self.model_name)
        print('Model %s loaded!' % (self.model_name))
        self.model = eval(self.model_name + '(self.args)')
        self.model.to(self.device)

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-8, weight_decay=self.weight_decay)

    def get_loss_function(self):
        def bpr_loss(pos_scores, neg_scores):
            return -torch.log(1e-24 + torch.sigmoid(pos_scores - neg_scores)).mean()
        
        loss_dict = {
            'bpr': bpr_loss,
            'bce': nn.BCEWithLogitsLoss(),
            'mse': nn.MSELoss()
        }
        self.loss_function = loss_dict[self.loss]

    def execute(self):
        # write args
        perf_str = str(self.args)
        with open(self.base_path + 'stats.txt','a') as f:
            f.write(perf_str+"\n")

        self.model, self.start_epoch = self.restore_checkpoint(self.model, self.base_path, self.device) # restore the checkpoint

        start_time = time.time()
        # train the model if not test only
        if not self.test_only:
            print("start training")
            self.train()
            # test the model
            print("start testing")
            self.model = self.restore_certain_checkpoint(self.best_valid_epoch, self.model, self.base_path, self.device)
        end_time = time.time()

        self.model.eval() # evaluate the best model
        print_str = "The best epoch is % d, total training cost is %.1f" % (max(self.best_valid_epoch, self.start_epoch), end_time - start_time)
        with open(self.base_path +'stats.txt', 'a') as f:
            f.write(print_str + "\n")

        print('Validation Phase:')
        stop_metric = self.evaluate(self.model, self.data.valid_data, self.device, name='valid')

        print('Test Phase:')
        _ = self.evaluate(self.model, self.data.test_data, self.device, name='test')

        # self.recommend_top_k()
        # self.document_hyper_params_results(self.base_path, n_rets)

    def train(self) -> None:
        # TODO
        self.total_step=0
        self.stop_metric_max = 0
        self.best_epoch = 0
        self.current_patience = 0

        self.stop_flag = False
        for epoch in range(self.start_epoch, self.max_epoch):
            # print(self.model.embed_user.weight)
            if self.stop_flag: # early stop
                break
            # All models
            t1=time.time()
            losses = self.train_one_epoch(epoch) # train one epoch
            t2=time.time()
            self.document_running_loss(losses, epoch, t2-t1) # report the loss
            if (epoch + 1) % self.verbose == 0: # evaluate the model
                self.eval_and_check_early_stop(epoch)

    def train_one_epoch(self, epoch):
        # raise NotImplementedError
        num_rows=self.data.train_data.shape[0]
        num_batches=int(num_rows/self.batch_size)

        running_loss = 0
        pbar = tqdm(enumerate(range(num_batches)), mininterval=2, total = num_batches)
        for batch_i, index in pbar:
            batch = self.data.train_data.sample(n=self.batch_size).to_dict()
            seq = list(batch['seq'].values())
            len_seq = list(batch['len_seq'].values())
            target=list(batch['next'].values())
            target_neg = []
            for index in range(self.batch_size):
                neg=np.random.randint(self.item_num)
                while neg==target[index]:
                    neg = np.random.randint(self.item_num)
                target_neg.append(neg)

            self.optimizer.zero_grad()

            seq, len_seq, target, target_neg = torch.LongTensor(seq), torch.LongTensor(len_seq), torch.LongTensor(target), torch.LongTensor(target_neg)
            seq, target, len_seq, target_neg = seq.to(self.device), target.to(self.device), len_seq.to(self.device), target_neg.to(self.device)

            model_output = self.model.forward(seq, len_seq)
            model_output = F.elu(model_output) + 1

            # print(target.shape, target_neg.shape, model_output.shape)
            target = target.view(self.batch_size, 1)
            target_neg = target_neg.view(self.batch_size, 1)

            pos_scores = torch.gather(model_output, 1, target)
            neg_scores = torch.gather(model_output, 1, target_neg)

            pos_labels = torch.ones((self.batch_size, 1))
            neg_labels = torch.zeros((self.batch_size, 1))

            scores = torch.cat((pos_scores, neg_scores), 0)
            labels = torch.cat((pos_labels, neg_labels), 0)
            labels = labels.to(self.device)

            loss = -torch.log(1e-24 + torch.sigmoid(pos_scores - neg_scores)).mean()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.detach().item()

        return [running_loss/num_batches]

    def document_running_loss(self, losses:list, epoch, t_one_epoch, prefix=""):
        loss_str = ', '.join(['%.5f']*len(losses)) % tuple(losses)
        perf_str = prefix + 'Epoch %d [%.1fs]: train==[' % (
                epoch, t_one_epoch) + loss_str + ']'
        with open(self.base_path + 'stats.txt','a') as f:
                f.write(perf_str+"\n")

    def eval_and_check_early_stop(self, epoch):
        self.model.eval()
        print('Validation Phase:')
        stop_metric = self.evaluate(self.model, self.data.valid_data, self.device, name='valid')
        # stop_metric = self.evaluate(self.model, self.data.valid_data, self.device, name='valid')
        # self.model.train()
        # stop_metric = self.evaluate(self.model, self.data.valid_data, self.device, name='valid')
        # stop_metric = self.evaluate(self.model, self.data.valid_data, self.device, name='valid')
        # print('TEST PHRASE:')
        # _ = self.evaluate(self.model, self.data.test_data, self.device, name='test')

        if stop_metric > self.stop_metric_max:
            self.stop_metric_max = stop_metric
            self.best_valid_epoch = epoch
            self.current_patience = 0
            checkpoint_buffer = self.save_checkpoint(self.model, epoch, self.base_path, self.checkpoint_buffer, 5)
        
        else:
            self.current_patience += 1
            if(self.current_patience >= self.patience):
                self.stop_flag = True
                print('Early stop at epoch %d' % epoch)
        
        print('BEST EPOCH:{}'.format(self.best_valid_epoch))
        self.model.train()
    
    @torch.no_grad()
    def evaluate(self, model, eval_sessions, device, name='valid'):
        # TODO 改candidate形式
        t1 = time.time()
        eval_ids = eval_sessions.session_id.unique()
        groups = eval_sessions.groupby('session_id')
        batch, evaluated, total_clicks, total_purchase= 100, 0, 1.0, 0.0

        total_reward = [0, 0, 0, 0]
        hit_clicks = [0, 0, 0, 0]
        ndcg_clicks = [0, 0, 0, 0]
        hit_purchase = [0, 0, 0, 0]
        ndcg_purchase = [0, 0, 0, 0]
        while evaluated<len(eval_ids):
            states, len_states, actions, rewards = [], [], [], []
            for i in range(batch):
                id=eval_ids[evaluated]
                group=groups.get_group(id)
                history=[]
                for index, row in group.iterrows():
                    state=list(history)
                    state = [int(i) for i in state]
                    len_states.append(self.seq_size if len(state)>=self.seq_size else 1 if len(state)==0 else len(state))
                    state=pad_history(state,self.seq_size,self.item_num)
                    states.append(state)
                    action=row['item_id']
                    try:
                        is_buy=row['t_read']
                    except:
                        is_buy=row['time']
                    reward = 1 if is_buy >0 else 0
                    if is_buy>0:
                        total_purchase+=1.0
                    else:
                        total_clicks+=1.0
                    actions.append(action)
                    rewards.append(reward)
                    history.append(row['item_id'])
                evaluated+=1
                if evaluated >= len(eval_ids):
                    break

            states = np.array(states)
            states = torch.LongTensor(states)
            states = states.to(device)

            prediction = model.forward_eval(states, np.array(len_states))
            prediction = prediction.cpu()
            prediction = prediction.detach().numpy()
            sorted_list=np.argsort(prediction)
            calculate_hit(sorted_list,self.topk,actions,rewards,self.reward_click,total_reward,hit_clicks,ndcg_clicks,hit_purchase,ndcg_purchase)

        hr_list = []
        ndcg_list = []
        # Divide by number of users
        for i in range(len(self.topk)):
            hr_purchase=hit_purchase[i]/total_purchase
            ng_purchase=ndcg_purchase[i]/total_purchase

            hr_list.append(hr_purchase)
            ndcg_list.append(ng_purchase[0,0])

            if i == 1:
                hr_20 = hr_purchase

        n_ret = {f"hr@{self.topk[0]}": hr_list[0], f"ndcg@{self.topk[0]}": ndcg_list[0], 
                f"hr@{self.topk[1]}": hr_list[1], f"ndcg@{self.topk[1]}": ndcg_list[1], 
                f"hr@{self.topk[2]}": hr_list[2], f"ndcg@{self.topk[2]}": ndcg_list[2]}

        perf_str = name+':{}'.format(n_ret)
        print(perf_str)
        with open(self.base_path + 'stats.txt', 'a') as f:
            f.write(perf_str + "\n")
        t2 = time.time()
        t_eval = t2 - t1
        print('[{}] Eval time:{:.1f}s'.format(name, t_eval))
        return hr_20

    def save_checkpoint(self, model, epoch, checkpoint_dir, buffer, max_to_keep=5):
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }

        filename = os.path.join(checkpoint_dir, 'epoch={}.checkpoint.pth.tar'.format(epoch))
        torch.save(state, filename)
        buffer.append(filename)
        if len(buffer)>max_to_keep:
            os.remove(buffer[0])
            del(buffer[0])

        return buffer

    def restore_checkpoint(self, model, checkpoint_dir, device, force=False, pretrain=False):
        """
        If a checkpoint exists, restores the PyTorch model from the checkpoint.
        Returns the model and the current epoch.
        """
        cp_files = [file_ for file_ in os.listdir(checkpoint_dir)
                    if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]

        if not cp_files:
            print('No saved model parameters found')
            if force:
                raise Exception("Checkpoint not found")
            else:
                return model, 0,

        epoch_list = []

        regex = re.compile(r'\d+')

        for cp in cp_files:
            epoch_list.append([int(x) for x in regex.findall(cp)][0])

        epoch = max(epoch_list)

        if not force:
            print("Which epoch to load from? Choose in range [0, {})."
                .format(epoch), "Enter 0 to train from scratch.")
            print(">> ", end = '')
            # inp_epoch = int(input())

            if self.args.clear_checkpoints:
                print("Clear checkpoint")
                clear_checkpoint(checkpoint_dir)
                return model, 0,

            inp_epoch = epoch
            if inp_epoch not in range(epoch + 1):
                raise Exception("Invalid epoch number")
            if inp_epoch == 0:
                print("Checkpoint not loaded")
                clear_checkpoint(checkpoint_dir)
                return model, 0,
        else:
            print("Which epoch to load from? Choose in range [0, {}).".format(epoch))
            inp_epoch = int(input())
            if inp_epoch not in range(0, epoch):
                raise Exception("Invalid epoch number")

        filename = os.path.join(checkpoint_dir,
                                'epoch={}.checkpoint.pth.tar'.format(inp_epoch))

        print("Loading from checkpoint {}?".format(filename))

        checkpoint = torch.load(filename, map_location = str(device))
        # print("finish load")

        try:
            if pretrain:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint['state_dict'])
            print("=> Successfully restored checkpoint (trained for {} epochs)"
                .format(checkpoint['epoch']))
        except:
            print("=> Checkpoint not successfully restored")
            raise

        return model, inp_epoch

    def restore_certain_checkpoint(self, epoch, model, checkpoint_dir, device):
        """
        Restore the best performance checkpoint
        """

        filename = os.path.join(checkpoint_dir,
                                'epoch={}.checkpoint.pth.tar'.format(epoch))

        print("Loading from checkpoint {}?".format(filename))

        checkpoint = torch.load(filename, map_location = str(device))

        model.load_state_dict(checkpoint['state_dict'])
        print("=> Successfully restored checkpoint (trained for {} epochs)"
            .format(checkpoint['epoch']))

        return model