import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from .base.utils import *

from .base.abstract_model import AbstractModel
from .base.abstract_RS import AbstractRS
from .base.abstract_data import AbstractData

from tqdm import tqdm


class AdvInfoNCE_RS(AbstractRS):
    def __init__(self, args, special_args) -> None:
        super().__init__(args, special_args)
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1
        self.adv_interval = args.adv_interval
        self.adv_epochs = args.adv_epochs
        self.adv_lr = args.adv_lr
        self.warm_up_epochs = args.warm_up_epochs
        self.eta_epochs = args.eta_epochs

        self.current_eta = 0 # adversarial training epochs
        self.eta_per_epoch = {} # document the disturbance strength for each user

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam([param for param in self.model.parameters() if param.requires_grad == True], lr=self.lr)
        self.adv_optimizer = torch.optim.Adam([param for param in self.model.parameters() if param.requires_grad == False], lr=self.adv_lr)

    def train_one_epoch(self, epoch):
        # adversarial train the the weights of negative items
        if (epoch + 3)  % self.adv_interval == 0 and self.current_eta < self.eta_epochs and epoch > self.warm_up_epochs:
            self.adversarial_training()

        running_loss, running_mf_loss, running_reg_loss, num_batches = 0, 0, 0, 0
        
        pbar = tqdm(enumerate(self.data.train_loader), mininterval=2, total = len(self.data.train_loader))
        for batch_i, batch in pbar:          

            batch = [x.cuda(self.device) for x in batch]
            users, pos_items, users_pop, pos_items_pop  = batch[0], batch[1], batch[2], batch[3]

            if self.args.infonce == 0 or self.args.neg_sample != -1:
                neg_items = batch[4]
                neg_items_pop = batch[5]

            self.model.train()
            mf_loss, reg_loss, reg_loss_prob, eta_u_, p_negative = self.model(users, pos_items, neg_items)
            loss = mf_loss + reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.detach().item()
            running_reg_loss += reg_loss.detach().item()
            running_mf_loss += mf_loss.detach().item()
            num_batches += 1
        return [running_loss/num_batches, running_mf_loss/num_batches, running_reg_loss/num_batches]
    
    def adversarial_training(self):
        print("start adversarial training...")
        print(f"current eta: {self.current_eta}/{self.eta_epochs}")
        running_loss, running_mf_loss, running_reg_loss, num_batches = 0, 0, 0, 0
        self.model.freeze_prob(False)
        for epoch_adv in range(self.adv_epochs):
            print(f"current adv epoch: {epoch_adv}/{self.adv_epochs}")

            adv_pbar = tqdm(enumerate(self.data.train_loader), mininterval=2, total = len(self.data.train_loader))

            t1 = time.time()
            eta_u = {}
            for batch_i, batch in adv_pbar: 
                batch = [x.cuda(self.device) for x in batch]
                users, pos_items, users_pop, pos_items_pop  = batch[0], batch[1], batch[2], batch[3]

                if self.args.infonce == 0 or self.args.neg_sample != -1:
                    neg_items = batch[4]
                    neg_items_pop = batch[5]

                self.model.train()
                mf_loss, reg_loss, reg_loss_prob, eta_u_, p_negative = self.model(users, pos_items, neg_items)
                loss = reg_loss_prob - mf_loss # maximize the mf_loss 

                for u, kl_ds_ in eta_u_.items():
                    if u in eta_u.keys():
                        eta_u[u].extend(kl_ds_)
                    else:
                        eta_u[u] = kl_ds_

                self.adv_optimizer.zero_grad()
                loss.backward()
                self.adv_optimizer.step()

                running_loss += loss.detach().item()
                running_reg_loss += reg_loss_prob.detach().item()
                running_mf_loss += mf_loss.detach().item()
                num_batches += 1
            
            t2 = time.time()
            self.document_running_loss([running_loss/num_batches, running_mf_loss/num_batches, running_reg_loss/num_batches], self.current_eta, t2-t1, "Adv")
            self.current_eta += 1

            self.document_adv_weights(p_negative.cpu().detach().numpy(), neg_items_pop.cpu().detach().numpy())
            self.document_eta(eta_u)

        self.model.freeze_prob(True)
    
    def document_adv_weights(self, p_negative, neg_items_pop):
        p_negative_sorted = np.zeros(p_negative.shape)
        for i in range(len(neg_items_pop)):
            p_negative_sorted[i] = p_negative[i][neg_items_pop[i].argsort()]   

        save_fig_path = self.base_path + "/distribution"
        ensureDir(save_fig_path)
        # after sorting
        idxs = np.random.randint(0, len(neg_items_pop), 9)
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))

        # random sample 9 users
        fig.suptitle('p_negative_sorted ')
        for i in range(3):
            for j in range(3):
                idx = idxs[i*3+j]
                axes[i, j].bar(np.arange(0,len(p_negative_sorted[idx]),1), p_negative_sorted[idx], align='center', alpha=0.5)
        plt.savefig(save_fig_path + f"/{self.current_eta}_p_negative_sorted.png")
        plt.close()

        # mean
        mean_p_negative = np.mean(p_negative_sorted, axis=0)
        plt.bar(np.arange(0,len(mean_p_negative),1), mean_p_negative, align='center', alpha=0.5)
        plt.title("mean_p_negative_sorted ")
        plt.savefig(save_fig_path + f"/{self.current_eta}_mean_p_negative_sorted.png") # save
        plt.close()

    def document_eta(self, eta_u):
        
        for u, kl_ds in eta_u.items():
            if u not in self.eta_per_epoch.keys():
                self.eta_per_epoch[u] = [np.mean(kl_ds)]
            else:
                self.eta_per_epoch[u].append(np.mean(kl_ds))
            # self.eta_per_epoch[u] = self.eta_per_epoch.get(u, []).extend([np.mean(kl_ds)])
        # print(self.eta_per_epoch)
        if 'mean' not in self.eta_per_epoch.keys():
            self.eta_per_epoch['mean'] = [np.mean([np.mean(v) for v in eta_u.values()])]
        else:
            self.eta_per_epoch['mean'].append(np.mean([np.mean(v) for v in eta_u.values()]))
        print(np.mean([np.mean(v) for v in eta_u.values()]))

        with open(self.base_path + 'stats.txt', 'a') as f:
            f.write("mean_eta" + "\t" + str(self.eta_per_epoch['mean']) + "\n")

class AdvInfoNCE(AbstractModel):
    def __init__(self, args, data) -> None:
        super().__init__(args, data)
        self.tau = args.tau
        self.k_neg = args.k_neg
        self.w_emb_dim = args.w_embed_size
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1
        self.model_version = args.model_version

        if(self.model_version == "mlp"): # MLP version
            self.w_emb_dim = 4
            self.u_mlp = nn.Sequential(
                nn.Linear(self.emb_dim, self.w_emb_dim),
                nn.ReLU())
            self.i_mlp = nn.Sequential(
                nn.Linear(self.emb_dim, self.w_emb_dim),
                nn.ReLU())
        else: # Embedding version
            self.embed_user_p = nn.Embedding(self.data.n_users, self.w_emb_dim)
            self.embed_item_p = nn.Embedding(self.data.n_items, self.w_emb_dim)
            nn.init.xavier_normal_(self.embed_user_p.weight)
            nn.init.xavier_normal_(self.embed_item_p.weight)
        self.freeze_prob(True)
    
    def forward(self, users, pos_items, neg_items):

        #@ Main Branch
        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        #@ Weight Branch
        if(self.model_version == "mlp"):
            users_p_emb = self.u_mlp(userEmb0.detach())
            neg_p_emb = self.i_mlp(negEmb0.detach())
        else:
            users_p_emb = self.embed_user_p(users)
            neg_p_emb = self.embed_item_p(neg_items)

        s_negative = torch.matmul(torch.unsqueeze(users_p_emb, 1), 
                                    neg_p_emb.permute(0, 2, 1)).squeeze(dim=1)
        p_negative = torch.softmax(s_negative, dim=1) # score for negative samples

        # main branch
        # use cosine similarity
        if(self.train_norm):
            users_emb = F.normalize(users_emb, dim = -1)
            pos_emb = F.normalize(pos_emb, dim = -1)
            neg_emb = F.normalize(neg_emb, dim = -1)

        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), 
                                       neg_emb.permute(0, 2, 1)).squeeze(dim=1)

        numerator = torch.exp(pos_ratings / self.tau)
        denominator = numerator + self.k_neg * int(p_negative.shape[1]) * torch.sum(torch.exp(neg_ratings / self.tau)*p_negative, dim = 1) #@ multiply with N

        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 ** torch.norm(negEmb0)
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        reg_neg_prob = 0.5 * torch.norm(users_p_emb) ** 2 + 0.5 * torch.norm(neg_p_emb) ** 2
        reg_neg_prob = reg_neg_prob / self.batch_size
        reg_loss_prob = self.decay * regularizer

        #@ calculate eta (KL divergence)
        kl_d = (p_negative*torch.log(p_negative/(1/self.neg_sample))).cpu().detach().numpy()
        kl_d = np.sum(kl_d, axis=1)
        # print(kl_d.shape, type(kl_d))
        # print(max(kl_d))
        eta_u_ = {}
        for idx, u in enumerate(list(users.cpu().detach().numpy())):
            kl_d_u = kl_d[idx]
            if u not in eta_u_.keys():
                eta_u_[u] = [kl_d_u]
            else:
                eta_u_[u].append(kl_d_u)

        return ssm_loss, reg_loss, reg_loss_prob, eta_u_, p_negative
    
    def freeze_prob(self, flag):
        if(self.model_version == "mlp"):
            if flag:
                for param in self.u_mlp.parameters():
                    param.requires_grad = False
                for param in self.i_mlp.parameters():
                    param.requires_grad = False
                self.embed_user.requires_grad_(True)
                self.embed_item.requires_grad_(True)
            else:
                for param in self.u_mlp.parameters():
                    param.requires_grad = True
                for param in self.i_mlp.parameters():
                    param.requires_grad = True
                self.embed_user.requires_grad_(False)
                self.embed_item.requires_grad_(False)
        else:
            if flag:
                self.embed_user_p.requires_grad_(False)
                self.embed_item_p.requires_grad_(False)
                self.embed_user.requires_grad_(True)
                self.embed_item.requires_grad_(True)
            else:
                self.embed_user_p.requires_grad_(True)
                self.embed_item_p.requires_grad_(True)
                self.embed_user.requires_grad_(False)
                self.embed_item.requires_grad_(False)