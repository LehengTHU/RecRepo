import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from .base.abstract_model import AbstractModel
from .base.abstract_RS import AbstractRS
from .base.abstract_data import AbstractData, helper_load, helper_load_train
from tqdm import tqdm

from .base.evaluator import ProxyEvaluator
from .base.utils import *
from .base.model_utils import *

from torch.utils.data import Dataset, DataLoader

from functools import partial

from reckit import randint_choice
import bisect
import random as rd


class PWLayer(nn.Module):
    """Single Parametric Whitening Layer
    """
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)

class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor
    """
    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training) # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)

class PLMEmb:
    def __init__(self, args):
        self.item_drop_ratio = args.item_drop_ratio

    def __call__(self, dataset, interaction):
        '''Sequence augmentation and PLM embedding fetching
        '''
        item_seq_len = interaction['item_length']
        item_seq = interaction['item_id_list']

        plm_embedding = dataset.plm_embedding
        item_emb_seq = plm_embedding(item_seq)
        pos_item_id = interaction['item_id']
        pos_item_emb = plm_embedding(pos_item_id)

        mask_p = torch.full_like(item_seq, 1 - self.item_drop_ratio, dtype=torch.float)
        mask = torch.bernoulli(mask_p).to(torch.bool)

        # Augmentation
        seq_mask = item_seq.eq(0).to(torch.bool)
        mask = torch.logical_or(mask, seq_mask)
        mask[:,0] = True
        drop_index = torch.cumsum(mask, dim=1) - 1

        item_seq_aug = torch.zeros_like(item_seq).scatter(dim=-1, index=drop_index, src=item_seq)
        item_seq_len_aug = torch.gather(drop_index, 1, (item_seq_len - 1).unsqueeze(1)).squeeze() + 1
        item_emb_seq_aug = plm_embedding(item_seq_aug)

        interaction.update({
            'item_emb_list': item_emb_seq,
            'pos_item_emb': pos_item_emb,
            'item_id_list_aug': item_seq_aug,
            'item_length_aug': item_seq_len_aug,
            'item_emb_list_aug': item_emb_seq_aug,
        })

        return interaction


class UniSRec_RS(AbstractRS):
    def __init__(self, args, special_args) -> None:
        super().__init__(args, special_args)
        self.transform = PLMEmb(args)
        
    def get_interaction(self, dataset, batch):
        # batch = [x.cuda(self.device) for x in batch]
        users, pos_items, item_seq = batch[0], batch[1], batch[2]
        interaction = {}
        interaction['item_id'] = pos_items
        interaction['item_id_list'] = item_seq
        # item_length 为B,N, N为item_seq的shape[1]
        interaction['item_length'] = torch.tensor([item_seq.shape[1]]*item_seq.shape[0])
        
        interaction = self.transform(dataset, interaction)
        for k, v in interaction.items():
            interaction[k] = v.to(self.device)
            # print(k, v.shape)
        return interaction
        # interaction['item_emb_list'] = item_seq_embs

    def train_one_epoch(self, epoch):
        running_loss, num_batches = 0, 0

        pbar = tqdm(enumerate(self.data.train_loader), mininterval=2, total = len(self.data.train_loader))
        for batch_i, batch in pbar:          
            interaction = self.get_interaction(self.data, batch)

            self.model.train()
            loss = self.model.calculate_loss(interaction)
            # print(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.detach().item()
            num_batches += 1
            # break
        return [running_loss/num_batches]

class UniSRec_Data(AbstractData):
    def __init__(self, args):
        super().__init__(args)

    
    def _convert_sp_mat_to_sp_tensor(self, X):
        return None

    def getSparseGraph(self):
        return None
    
    def add_special_model_attr(self, args):
        self.max_seq_length = args.max_seq_length
        self.load_plm_embedding(args)
        
    def load_plm_embedding(self, args):
        loading_path = args.data_path + args.dataset + '/item_info/'
        weight = np.load(loading_path + 'item_cf_embeds_bert_array.npy') # bert
        plm_embedding = nn.Embedding(weight.shape[0], weight.shape[1], padding_idx=0)
        plm_embedding.weight.requires_grad = False
        plm_embedding.weight.data.copy_(torch.from_numpy(weight))
        self.plm_embedding = plm_embedding
        # self.plm_embedding = plm_embedding.to(self.device)
        # self.plm_embedding = self.plm_embedding
    
    def get_dataloader(self):
        # return super().get_dataloader()
        self.train_data = TrainDataset(self.model_name, self.users, self.train_user_list,\
                                        self.n_observations, self.n_items, self.items, self.plm_embedding, self.max_seq_length)

        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)


class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, model_name, users, train_user_list, \
                n_observations, n_items, items, plm_embedding, max_seq_length):
        self.model_name = model_name
        self.users = users
        self.items = items
        self.train_user_list = train_user_list
        self.n_observations = n_observations
        self.n_items = n_items
        self.plm_embedding = plm_embedding
        self.max_seq_length = max_seq_length
        
    def __getitem__(self, index):

        index = index % len(self.users)
        user = self.users[index]
        if self.train_user_list[user] == []:
            pos_items = 0
        else:
            pos_item = rd.choice(self.train_user_list[user])

        # 从self.train_user_list[user]中放回抽样20个item
        item_seq = np.random.choice(self.train_user_list[user], self.max_seq_length)
        # item_seq_embs = self.plm_embedding(torch.tensor(item_seq))
        
        return user, pos_item, item_seq

    def __len__(self):
        return self.n_observations

class UniSRec(nn.Module):
    def __init__(self, args, dataset):
        super().__init__()
        self.data = dataset
        self.device = torch.device(args.cuda)

        self.train_stage = args.train_stage
        self.temperature = args.temperature
        self.lam = args.lambda_
        

        assert self.train_stage in [
            'pretrain', 'inductive_ft', 'transductive_ft'
        ], f'Unknown train stage: [{self.train_stage}]'

        if self.train_stage in ['pretrain', 'inductive_ft']:
            self.item_embedding = None
            # for `transductive_ft`, `item_embedding` is defined in SASRec base model
        if self.train_stage in ['inductive_ft', 'transductive_ft']:
            # `plm_embedding` in pre-train stage will be carried via dataloader
            self.plm_embedding = copy.deepcopy(dataset.plm_embedding)

        self.moe_adaptor = MoEAdaptorLayer(
            args.n_exps,
            [args.plm_hidden_size, args.hidden_size],
            args.adaptor_dropout_prob
        )
        self.init_sasrec(args)
        
    def init_sasrec(self, args):
        self.max_seq_length = args.max_seq_length
        self.hidden_size = args.hidden_size
        self.hidden_dropout_prob = args.hidden_dropout_prob
        self.layer_norm_eps = args.layer_norm_eps
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.trm_encoder = TransformerEncoder(
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            hidden_size=args.hidden_size,
            inner_size=args.inner_size,
            hidden_dropout_prob=args.hidden_dropout_prob,
            attn_dropout_prob=args.attn_dropout_prob,
            hidden_act=args.hidden_act,
            layer_norm_eps=args.layer_norm_eps,
        )
        
    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask
    
    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def forward(self, item_seq, item_emb, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        if self.train_stage == 'transductive_ft':
            input_emb = input_emb + self.item_embedding(item_seq)
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb) # B N D

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def seq_item_contrastive_task(self, seq_output, same_pos_id, interaction):
        pos_items_emb = self.moe_adaptor(interaction['pos_item_emb'])
        pos_items_emb = F.normalize(pos_items_emb, dim=1)

        pos_logits = (seq_output * pos_items_emb).sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, pos_items_emb.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean()

    def seq_seq_contrastive_task(self, seq_output, same_pos_id, interaction):
        item_seq_aug = interaction['item_id_list' + '_aug']
        item_seq_len_aug = interaction['item_length' + '_aug']
        item_emb_list_aug = self.moe_adaptor(interaction['item_emb_list_aug'])
        seq_output_aug = self.forward(item_seq_aug, item_emb_list_aug, item_seq_len_aug)
        seq_output_aug = F.normalize(seq_output_aug, dim=1)

        pos_logits = (seq_output * seq_output_aug).sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, seq_output_aug.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean()

    def pretrain(self, interaction):
        item_seq = interaction['item_id_list']
        item_seq_len = interaction['item_length']
        item_emb_list = self.moe_adaptor(interaction['item_emb_list'])
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        seq_output = F.normalize(seq_output, dim=1)

        # Remove sequences with the same next item
        pos_id = interaction['item_id']
        same_pos_id = (pos_id.unsqueeze(1) == pos_id.unsqueeze(0))
        same_pos_id = torch.logical_xor(same_pos_id, torch.eye(pos_id.shape[0], dtype=torch.bool, device=pos_id.device))

        loss_seq_item = self.seq_item_contrastive_task(seq_output, same_pos_id, interaction)
        loss_seq_seq = self.seq_seq_contrastive_task(seq_output, same_pos_id, interaction)
        loss = loss_seq_item + self.lam * loss_seq_seq
        return loss

    def calculate_loss(self, interaction):
        if self.train_stage == 'pretrain':
            return self.pretrain(interaction)

        # Loss for fine-tuning
        item_seq = interaction['item_id_list']
        item_seq_len = interaction['item_length']
        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        test_item_emb = self.moe_adaptor(self.plm_embedding.weight)
        if self.train_stage == 'transductive_ft':
            test_item_emb = test_item_emb + self.item_embedding.weight

        seq_output = F.normalize(seq_output, dim=1)
        test_item_emb = F.normalize(test_item_emb, dim=1)

        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        pos_items = interaction[self.POS_ITEM_ID]
        loss = self.loss_fct(logits, pos_items)
        return loss

    @torch.no_grad()
    def predict(self, users, items=None):
        item_seqs, item_max_lens = [], [self.data.max_seq_length]*len(users)
        for u in users:
            item_seq = np.random.choice(self.data.train_user_list[u], self.data.max_seq_length)
            item_seqs.append(item_seq)
            # print(u)
        item_seq = torch.tensor(np.array(item_seqs)).to(self.device)
        item_seq_len = torch.tensor(np.array(item_max_lens)).to(self.device)
        plm_embedding = self.data.plm_embedding
        plm_embedding.to(self.device)
        # print(plm_embedding.weight.data.device)
        item_emb_list = self.moe_adaptor(plm_embedding(item_seq))
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        test_items_emb = self.moe_adaptor(plm_embedding.weight)
        if self.train_stage == 'transductive_ft':
            test_items_emb = test_items_emb + self.item_embedding.weight

        seq_output = F.normalize(seq_output, dim=-1)
        test_items_emb = F.normalize(test_items_emb, dim=-1)
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        plm_embedding.to('cpu')
        return scores.cpu().detach().numpy()