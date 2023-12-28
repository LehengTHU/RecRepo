import numpy as np
import pandas as pd
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from .base.utils import *
from .base.model_utils import *
from .base.abstract_RS import AbstractRS

class GRU4Rec_RS(AbstractRS):
    def __init__(self, args) -> None:
        super().__init__(args)


class GRU4Rec(nn.Module):
    # def __init__(self, hidden_size, item_num, state_size, gru_layers=1):
    def __init__(self, args):
        super(GRU4Rec, self).__init__()
        self.hidden_size = args.hidden_size
        self.item_num = args.item_num
        self.state_size = args.state_size
        self.item_embeddings = nn.Embedding(
            num_embeddings=args.item_num + 1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=args.gru_layers,
            batch_first=True
        )
        self.s_fc = nn.Linear(self.hidden_size, self.item_num)

    def forward(self, states, len_states):
        # Supervised Head
        emb = self.item_embeddings(states)
        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(emb, len_states.cpu(), batch_first=True, enforce_sorted=False)
        emb_packed, hidden = self.gru(emb_packed)
        hidden = hidden.view(-1, hidden.shape[2])
        supervised_output = self.s_fc(hidden)
        return supervised_output

    def forward_eval(self, states, len_states):
        # Supervised Head
        emb = self.item_embeddings(states)
        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(emb, len_states, batch_first=True, enforce_sorted=False)
        emb_packed, hidden = self.gru(emb_packed)
        hidden = hidden.view(-1, hidden.shape[2])
        supervised_output = self.s_fc(hidden)

        return supervised_output

