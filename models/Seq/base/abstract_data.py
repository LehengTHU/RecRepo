import pandas as pd
import os

class AbstractData:
    def __init__(self, args):
        self.args = args
        self.data_directory = './data/' + args.dataset
        self.load_data()
        self.get_attributes()

    def load_data(self):
        self.train_data = pd.read_pickle(os.path.join(self.data_directory, 'train_data.df'))
        self.valid_data = pd.read_pickle(os.path.join(self.data_directory, 'val_sessions.df'))
        self.test_data = pd.read_pickle(os.path.join(self.data_directory, 'test_sessions.df'))
        self.data_statis = pd.read_pickle(os.path.join(self.data_directory, 'data_statis.df'))
    
    def get_attributes(self):
        self.seq_size = self.data_statis['seq_size'][0]  # the length of history to define the seq
        self.item_num = self.data_statis['item_num'][0]  # total number of items

        self.best_valid_epoch = 0