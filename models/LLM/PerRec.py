import os
import random
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from transformers import LlamaForCausalLM, LlamaTokenizer

from ..Seq.SASRec import SASRec
from ..Seq.Caser import Caser
from ..Seq.GRU4Rec import GRU4Rec
# from recommender.Dream import DreamRec, SinusoidalPositionEmbeddings
# import recommender.Dream

# from .base.data_utils import *
from .base.data_utils import *
from .base.utils import *
from .base.abstract_RS import AbstractRS
# from .base.abstract_model import AbstractModel

# from evaluator import *

class PerRec_RS(AbstractRS):
    def __init__(self, args) -> None:
        super().__init__(args)

class PerRec(nn.Module):
    # def __init__(self, hidden_size, item_num, state_size, num_filters, filter_sizes,
    #              dropout_rate):
    def __init__(self, args):
        print('hello')


print("finished importing")

