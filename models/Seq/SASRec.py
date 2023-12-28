import numpy as np
import pandas as pd
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from .base.utils import *
from .base.model_utils import *
from .base.abstract_RS import AbstractRS

class SASRec_RS(AbstractRS):
    def __init__(self, args) -> None:
        super().__init__(args)

class SASRec(nn.Module):
    # def __init__(self, hidden_size, item_num, state_size, dropout, device, num_heads=1):
    def __init__(self, args):
        super().__init__()
        # super(SASRec, self).__init__()
        self.state_size = args.state_size
        self.hidden_size = args.hidden_size
        self.item_num = int(args.item_num)
        self.dropout = nn.Dropout(args.dropout)
        # self.device = args.device
        self.device = torch.device(args.cuda)
        self.item_embeddings = nn.Embedding(
            num_embeddings=args.item_num + 1,
            embedding_dim=args.hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 1)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=args.state_size,
            embedding_dim=args.hidden_size
        )
        self.emb_dropout = nn.Dropout(args.dropout)
        self.ln_1 = nn.LayerNorm(args.hidden_size)
        self.ln_2 = nn.LayerNorm(args.hidden_size)
        self.ln_3 = nn.LayerNorm(args.hidden_size)
        self.mh_attn = MultiHeadAttention(args.hidden_size, args.hidden_size, args.num_heads, args.dropout)
        self.feed_forward = PositionwiseFeedForward(args.hidden_size, args.hidden_size, args.dropout)
        self.s_fc = nn.Linear(args.hidden_size, args.item_num)

    def forward(self, states, len_states):
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output

    def forward_eval(self, states, len_states):
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output
    
    def cacul_h(self, states, len_states):
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)

        return state_hidden
    
    def cacu_x(self, x):
        x = self.item_embeddings(x)

        return x