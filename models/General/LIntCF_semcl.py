import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base.abstract_model import AbstractModel
from .base.abstract_RS import AbstractRS
from .base.abstract_data import AbstractData
from tqdm import tqdm

from functools import partial

class LIntCF_semcl_RS(AbstractRS):
    def __init__(self, args, special_args) -> None:
        super().__init__(args, special_args)

    def train_one_epoch(self, epoch):
        running_loss, running_mf_loss, running_sem_cl_loss, num_batches = 0, 0, 0, 0

        pbar = tqdm(enumerate(self.data.train_loader), mininterval=2, total = len(self.data.train_loader))
        for batch_i, batch in pbar:          
            
            batch = [x.cuda(self.device) for x in batch]
            users, pos_items, users_pop, pos_items_pop  = batch[0], batch[1], batch[2], batch[3]

            if self.args.infonce == 0 or self.args.neg_sample != -1:
                neg_items = batch[4]
                neg_items_pop = batch[5]

            self.model.train()

            # loss = self.model(users, pos_items, neg_items)
            mf_loss, sem_cl_loss = self.model(users, pos_items, neg_items)
            loss = mf_loss + sem_cl_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.detach().item()
            running_mf_loss += mf_loss.detach().item()
            running_sem_cl_loss += sem_cl_loss.detach().item()
            num_batches += 1

        return [running_loss/num_batches, running_mf_loss/num_batches, running_sem_cl_loss/num_batches]

class LIntCF_semcl_Data(AbstractData):
    def __init__(self, args):
        super().__init__(args)
    
    def add_special_model_attr(self, args):
        loading_path = args.data_path + args.dataset + '/item_info/'
        self.item_cf_embeds = np.load(loading_path + 'item_cf_embeds_array.npy')
        self.item_aug_embeds = np.load(loading_path + 'item_aug_embeds_array.npy')

        def group_agg(group_data, embedding_dict, key='item_id'):
            ids = group_data[key].values
            # len_user = len(ids)
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
        # new_group_agg = partial(group_agg, embedding_dict=embedding_dict, key='item_id')
        item_cf_embeds_dict = {i:self.item_cf_embeds[i] for i in range(len(self.item_cf_embeds))}
        user_cf_embeds = groups.apply(group_agg, embedding_dict=item_cf_embeds_dict, key='item_id')
        user_cf_embeds_dict = user_cf_embeds.to_dict()
        user_cf_embeds_dict = dict(sorted(user_cf_embeds_dict.items(), key=lambda item: item[0]))

        self.user_cf_embeds = np.array(list(user_cf_embeds_dict.values()))

        # User Aug Embedding
        groups = pairs.groupby('user_id')
        # new_group_agg = partial(group_agg, embedding_dict=embedding_dict, key='item_id')
        item_aug_embeds_dict = {i:self.item_aug_embeds[i] for i in range(len(self.item_aug_embeds))}
        user_aug_embeds = groups.apply(group_agg, embedding_dict=item_aug_embeds_dict, key='item_id')
        user_aug_embeds_dict = user_aug_embeds.to_dict()
        user_aug_embeds_dict = dict(sorted(user_aug_embeds_dict.items(), key=lambda item: item[0]))

        self.user_aug_embeds = np.array(list(user_aug_embeds_dict.values()))


class LIntCF_semcl(AbstractModel):
    def __init__(self, args, data) -> None:
        super().__init__(args, data)
        self.tau = args.tau
        self.embed_size = args.hidden_size


        self.init_user_cf_embeds = data.user_cf_embeds
        self.init_item_cf_embeds = data.item_cf_embeds
        self.init_user_aug_embeds = data.user_aug_embeds
        self.init_item_aug_embeds = data.item_aug_embeds

        self.init_user_cf_embeds = torch.tensor(self.init_user_cf_embeds, dtype=torch.float32).cuda(self.device)
        self.init_item_cf_embeds = torch.tensor(self.init_item_cf_embeds, dtype=torch.float32).cuda(self.device)
        self.init_user_aug_embeds = torch.tensor(self.init_user_aug_embeds, dtype=torch.float32).cuda(self.device)
        self.init_item_aug_embeds = torch.tensor(self.init_item_aug_embeds, dtype=torch.float32).cuda(self.device)

        self.init_embed_shape = self.init_user_cf_embeds.shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(self.init_embed_shape, self.init_embed_shape//2),
            nn.LeakyReLU(),
            nn.Linear(self.init_embed_shape//2, self.embed_size)
        )
        self.proj = nn.Linear(self.embed_size, self.embed_size)

        self.temp_cl = args.temp_cl
        self.lambda_cl = args.lambda_cl
        self.eps = args.eps

    def init_embedding(self):
        pass


    def compute(self, perturbed=False):
        users_cf_emb = self.mlp(self.init_user_cf_embeds)
        items_cf_emb = self.mlp(self.init_item_cf_embeds)
        

        users_emb = users_cf_emb
        items_emb = items_cf_emb

        # print(users_emb.shape, items_emb.shape)
        all_emb = torch.cat([users_emb, items_emb])


        # noise on item side
        # if perturbed:
        #     random_noise = torch.rand_like(items_emb).cuda(self.device) # add noise
        #     items_emb_cl = items_emb + torch.sign(items_emb) * F.normalize(random_noise, dim=-1) * self.eps

        if perturbed:
            user_emb_aug = self.mlp(self.init_user_aug_embeds)
            items_emb_aug = self.mlp(self.init_item_aug_embeds)

        # if perturbed:
        #     random_noise = torch.rand_like(self.init_item_cf_embeds).cuda(self.device) # add noise
        #     items_emb_cl = self.init_item_cf_embeds + torch.sign(self.init_item_cf_embeds) * F.normalize(random_noise, dim=-1) * self.eps
        #     items_emb_cl = self.mlp(items_emb_cl)

        embs = [all_emb]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            # print(g_droped.device, all_emb.device)
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.data.n_users, self.data.n_items])

        if perturbed:
            return users, items, user_emb_aug, items_emb_aug
        
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

    def forward(self, users, pos_items, neg_items):

        all_users, all_items, user_emb_aug, items_emb_aug = self.compute(perturbed = True)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]


        # sem_cl_loss = self.lambda_cl * self.InfoNCE(all_items[pos_items], items_emb_cl[pos_items], self.temp_cl)
        # 计算mse
        # sem_cl_loss = self.lambda_cl * F.mse_loss(all_items[pos_items], items_emb_cl[pos_items])


        if(self.train_norm):
            users_emb = F.normalize(users_emb, dim = -1)
            pos_emb = F.normalize(pos_emb, dim = -1)
            neg_emb = F.normalize(neg_emb, dim = -1)


        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), 
                                       neg_emb.permute(0, 2, 1)).squeeze(dim=1)

        numerator = torch.exp(pos_ratings / self.tau)

        denominator = numerator + torch.sum(torch.exp(neg_ratings / self.tau), dim = 1)
        
        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))


        user_emb_aug = user_emb_aug[users]
        pos_emb_aug = items_emb_aug[pos_items]
        neg_emb_aug = items_emb_aug[neg_items]

        if(self.train_norm):
            user_emb_aug = F.normalize(user_emb_aug, dim = -1)
            pos_emb_aug = F.normalize(pos_emb_aug, dim = -1)
            neg_emb_aug = F.normalize(neg_emb_aug, dim = -1)

        pos_ratings_aug = torch.sum(user_emb_aug*pos_emb_aug, dim = -1)
        neg_ratings_aug = torch.matmul(torch.unsqueeze(user_emb_aug, 1), 
                                       neg_emb_aug.permute(0, 2, 1)).squeeze(dim=1)
        
        numerator_aug = torch.exp(pos_ratings_aug / self.tau)
        denominator_aug = numerator_aug + torch.sum(torch.exp(neg_ratings_aug / self.tau), dim = 1)

        sem_cl_loss = torch.mean(torch.negative(torch.log(numerator_aug/denominator_aug)))


        return ssm_loss, sem_cl_loss

    # @torch.no_grad()
    # def predict(self, users, items=None):
    #     if items is None:
    #         items = list(range(self.data.n_items))

    #     all_users, all_items = self.compute()

    #     users = all_users[torch.tensor(users).cuda(self.device)]
    #     items = all_items[torch.tensor(items).cuda(self.device)]
        
    #     if(self.pred_norm == True):
    #         users = F.normalize(users, dim = -1)
    #         items = F.normalize(items, dim = -1)

    #     items = torch.transpose(items, 0, 1)
    #     rate_batch = torch.matmul(users, items) # user * item

    #     return rate_batch.cpu().detach().numpy()


    @torch.no_grad()
    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.data.n_items))

        all_users, all_items, user_emb_aug, items_emb_aug = self.compute(perturbed = True)

        users_emb = all_users[torch.tensor(users).cuda(self.device)]
        items_emb = all_items[torch.tensor(items).cuda(self.device)]

        user_emb_aug = user_emb_aug[torch.tensor(users).cuda(self.device)]
        items_emb_aug = items_emb_aug[torch.tensor(items).cuda(self.device)]
        
        if(self.pred_norm == True):
            users_emb = F.normalize(users_emb, dim = -1)
            items_emb = F.normalize(items_emb, dim = -1)
            user_emb_aug = F.normalize(user_emb_aug, dim = -1)
            items_emb_aug = F.normalize(items_emb_aug, dim = -1)

        items_emb = torch.transpose(items_emb, 0, 1)
        rate_batch = torch.matmul(users_emb, items_emb) # user * item

        items_emb_aug = torch.transpose(items_emb_aug, 0, 1)
        rate_batch_aug = torch.matmul(user_emb_aug, items_emb_aug)

        # return rate_batch.cpu().detach().numpy()
        return rate_batch.cpu().detach().numpy() + rate_batch_aug.cpu().detach().numpy()


