import pandas as pd
import os
from .data_utils import *


class AbstractData:
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        self.data_directory = './data/LLM/' + args.dataset
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.load_data()
        self.get_additional_attributes()

    def load_data(self):
        if self.dataset == 'ml100k' or self.dataset == 'test_data':
            self.train_data = TrainingData(data4frame(), device=self.device)
            self.train_loader = DataLoader(self.train_data, batch_size=16, shuffle=True)

            self.test_data = TrainingData(data4frame_test(), device=self.device)
            self.test_loader = DataLoader(self.test_data, batch_size=32, shuffle=False)
        elif self.dataset == 'steam':
            self.train_data = TrainingData(data4frame_steam(), device=self.device)
            self.train_loader = DataLoader(self.train_data, batch_size=16, shuffle=True)

            self.test_data = TrainingData(data4frame_steam_test(), device=self.device)
            self.test_loader = DataLoader(self.test_data, batch_size=32, shuffle=False)
        else:
            raise ValueError("Dataset: {} is not supported".format(self.dataset))
    
    def get_additional_attributes(self):
    #     self.seq_size = self.data_statis['seq_size'][0]  # the length of history to define the seq
    #     self.item_num = self.data_statis['item_num'][0]  # total number of items

        # self.best_valid_epoch = 0
        return