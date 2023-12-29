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

from .base.data_utils import *
from .base.utils import *
from .base.abstract_RS import AbstractRS

from termcolor import colored, cprint

class PerRec_RS(AbstractRS):
    def __init__(self, args) -> None:
        super().__init__(args)

    def load_model(self):
        exec('from models.LLM.'+ self.model_name + ' import ' + self.model_name)
        print('Model %s loaded!' % (self.model_name))
        self.model = eval(self.model_name + '(self.args)')

class PerRec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.parse_args(args)
        self.load_llm()
        self.config_tuning()
    
    def parse_args(self, args):
        self.args = args
        self.dataset_name = args.dataset
        self.llm_path = args.llm_path
    
    def load_llm(self):
        self.llm = LlamaForCausalLM.from_pretrained(self.llm_path)
        self.tokenizer = LlamaTokenizer.from_pretrained(self.llm_path, device_map="auto")
        cprint('LLM loaded!', 'green', attrs=['bold'])

    def config_tuning(self):
        return


