import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from .base.abstract_model import AbstractModel
from .base.abstract_RS import AbstractRS
from .base.abstract_data import AbstractData
from tqdm import tqdm

def cal_rlmrec_infonce_loss(embeds1, embeds2, all_embeds2, temp=1.0):
    normed_embeds1 = embeds1 / torch.sqrt(1e-8 + embeds1.square().sum(-1, keepdim=True))
    normed_embeds2 = embeds2 / torch.sqrt(1e-8 + embeds2.square().sum(-1, keepdim=True))
    normed_all_embeds2 = all_embeds2 / torch.sqrt(1e-8 + all_embeds2.square().sum(-1, keepdim=True))
    nume_term = -(normed_embeds1 * normed_embeds2 / temp).sum(-1)
    deno_term = torch.log(torch.sum(torch.exp(normed_embeds1 @ normed_all_embeds2.T / temp), dim=-1))
    cl_loss = (nume_term + deno_term).sum()
    return cl_loss

class RLMRec_RS(AbstractRS):
    def __init__(self, args, special_args) -> None:
        super().__init__(args, special_args)
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1

    def train_one_epoch(self, epoch):
        running_loss, running_mf_loss, running_cl_loss, running_reg_loss, running_kd_loss, num_batches = 0, 0, 0, 0, 0, 0

        pbar = tqdm(enumerate(self.data.train_loader), mininterval=2, total = len(self.data.train_loader))
        for batch_i, batch in pbar:          
            
            batch = [x.cuda(self.device) for x in batch]
            users, pos_items, users_pop, pos_items_pop = batch[0], batch[1], batch[2], batch[3]

            if self.args.infonce == 0 or self.args.neg_sample != -1:
                neg_items = batch[4]
                neg_items_pop = batch[5]

            self.model.train()
            mf_loss, cl_loss, reg_loss, kd_loss = self.model(users, pos_items, neg_items)
            loss = mf_loss + cl_loss + reg_loss + kd_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.detach().item()
            running_mf_loss += mf_loss.detach().item()
            running_cl_loss += cl_loss.detach().item()
            running_reg_loss += reg_loss.detach().item()
            running_kd_loss += kd_loss.detach().item()
            num_batches += 1
        return [running_loss/num_batches, running_mf_loss/num_batches, running_cl_loss/num_batches, running_reg_loss/num_batches, running_kd_loss/num_batches]
    
class RLMRec_Data(AbstractData):
    def __init__(self, args):
        super().__init__(args)
    
    def add_special_model_attr(self, args):
        loading_path = args.data_path + args.dataset + '/item_info/'
        # self.item_sem_embeds = np.load(loading_path + 'item_cf_embeds_array.npy')
        self.item_sem_embeds = np.load(loading_path + 'item_cf_embeds_large3_array.npy')
        # self.item_sem_embeds = np.load(loading_path + 'item_cf_embeds_bert_array.npy')

        def group_agg(group_data, embedding_dict, key='item_id'):
            ids = group_data[key].values
            embeds = [embedding_dict[id] for id in ids]
            embeds = np.array(embeds)
            return embeds.mean(axis=0)

        # self.train_user_list
        pairs = []
        for u, v in self.train_user_list.items():
            for i in v:
                pairs.append((u, i))
        pairs = pd.DataFrame(pairs, columns=['user_id', 'item_id'])
        
        # User CF Embedding
        groups = pairs.groupby('user_id')
        item_sem_embeds_dict = {i:self.item_sem_embeds[i] for i in range(len(self.item_sem_embeds))}
        user_sem_embeds = groups.apply(group_agg, embedding_dict=item_sem_embeds_dict, key='item_id')
        user_sem_embeds_dict = user_sem_embeds.to_dict()
        user_sem_embeds_dict = dict(sorted(user_sem_embeds_dict.items(), key=lambda item: item[0]))

        self.user_sem_embeds = np.array(list(user_sem_embeds_dict.values()))

class RLMRec(AbstractModel):
    def __init__(self, args, data) -> None:
        super().__init__(args, data)
        self.temp_cl = args.temp_cl
        self.layer_cl = args.layer_cl
        self.lambda_cl = args.lambda_cl
        self.eps_XSimGCL = args.eps_XSimGCL
        
        self.kd_weight = args.kd_weight
        self.kd_temperature = args.kd_temperature
        
        self.embed_size = args.hidden_size
        self.init_user_sem_embeds = data.user_sem_embeds
        self.init_item_sem_embeds = data.item_sem_embeds

        self.init_user_sem_embeds = torch.tensor(self.init_user_sem_embeds, dtype=torch.float32).cuda(self.device)
        self.init_item_sem_embeds = torch.tensor(self.init_item_sem_embeds, dtype=torch.float32).cuda(self.device)

        self.init_embed_shape = self.init_user_sem_embeds.shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(self.init_embed_shape, self.init_embed_shape // 2),
            nn.LeakyReLU(),
            nn.Linear(self.init_embed_shape // 2, self.embed_size)
        )

    def compute(self, perturbed=False):
        users_emb = self.embed_user.weight
        items_emb = self.embed_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = []
        emb_cl = all_emb
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            if perturbed:
                random_noise = torch.rand_like(all_emb).cuda(self.device) # add noise
                all_emb += torch.sign(all_emb) * F.normalize(random_noise, dim=-1) * self.eps_XSimGCL
            embs.append(all_emb)
            if layer==self.layer_cl-1:
                emb_cl = all_emb
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)

        users, items = torch.split(light_out, [self.data.n_users, self.data.n_items])
        users_cl, items_cl = torch.split(emb_cl, [self.data.n_users, self.data.n_items]) # view of noise

        if perturbed:
            return users, items, users_cl, items_cl
        return users, items
    
    def InfoNCE(self, view1, view2, temperature, b_cos = True):
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score+10e-6)
        return torch.mean(cl_loss)
    
    def cal_cl_loss(self, idx, user_view1,user_view2,item_view1,item_view2):
        # 算的一个batch中的
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda(self.device)
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda(self.device)
        user_cl_loss = self.InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp_cl)
        item_cl_loss = self.InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp_cl)
        return user_cl_loss + item_cl_loss

    def forward(self, users, pos_items, neg_items):
        all_users, all_items, all_users_cl, all_items_cl = self.compute(perturbed=True)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        # contrastive loss
        cl_loss = self.lambda_cl * self.cal_cl_loss([users,pos_items], all_users, all_users_cl, all_items, all_items_cl)

        # main loss
        # use cosine similarity to calculate the scores
        if(self.train_norm == True):
            users_emb = F.normalize(users_emb, dim = -1)
            pos_emb = F.normalize(pos_emb, dim = -1)
            neg_emb = F.normalize(neg_emb, dim = -1)
        
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-6)
        mf_loss = torch.negative(torch.mean(maxi))

        # regularizer loss
        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        # RLMRec KD Loss
        usrprf_embeds = self.mlp(self.init_user_sem_embeds)
        itmprf_embeds = self.mlp(self.init_item_sem_embeds)

        users_emb_sem = usrprf_embeds[users]
        pos_emb_sem = itmprf_embeds[pos_items]
        neg_emb_sem = itmprf_embeds[neg_items]

        kd_loss = cal_rlmrec_infonce_loss(users_emb, users_emb_sem, users_emb_sem, self.kd_temperature) + \
                  cal_rlmrec_infonce_loss(pos_emb, pos_emb_sem, pos_emb_sem, self.kd_temperature) + \
                  cal_rlmrec_infonce_loss(neg_emb, neg_emb_sem, neg_emb_sem, self.kd_temperature)
        kd_loss /= users_emb_sem.shape[0]
        kd_loss *= self.kd_weight



        return mf_loss, cl_loss, reg_loss, kd_loss