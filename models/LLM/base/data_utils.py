import os
import pandas as pd
import random
import torch
from torch.utils.data import Dataset, DataLoader


def remove_1682(xx):
    x = xx[:]

    for i in range(10):
        try:
            x.remove(1682)
        except:
            break
    return x

def remove_3581(xx):
    x = xx[:]

    for i in range(10):
        try:
            x.remove(3581)
        except:
            break
    return x

def data4frame():
    movie_id_name = dict()

    with open(r'./data/LLM/test_data/u.item', 'r', encoding = "ISO-8859-1") as f:
        for l in f.readlines():
            ll = l.strip('\n').split('|')
            movie_name = ll[1][:-7]
            if movie_name[-5: ] == ', The':
                # movie_name = 'The ' + movie_name[:-5]
                movie_name = movie_name[:-5]
            movie_id_name[int(ll[0]) - 1] = movie_name

    train_data = pd.read_pickle('./data/LLM/test_data/train_data.df')
    train_data = train_data[train_data['len_seq'] >= 3]

    train_data['seq_unpad'] = train_data['seq'].apply(remove_1682)

    def index_to_movie(x):     
        movie_list = [movie_id_name[i] for i in x]
        movie_str = '::'.join(movie_list)
        return movie_str

    train_data['movie_seq'] = train_data['seq_unpad'].apply(index_to_movie)

    return train_data

def data4frame_test():
    movie_id_name = dict()

    with open(r'./data/LLM/test_data/u.item', 'r', encoding = "ISO-8859-1") as f:
        for l in f.readlines():
            ll = l.strip('\n').split('|')
            movie_name = ll[1][:-7]
            if movie_name[-5: ] == ', The':
                # movie_name = 'The ' + movie_name[:-5]
                movie_name = movie_name[:-5]
            movie_id_name[int(ll[0]) - 1] = movie_name


    train_data = pd.read_pickle('./data/LLM/test_data/Test_data.df')
    train_data = train_data[train_data['len_seq'] >= 3]


    train_data['seq_unpad'] = train_data['seq'].apply(remove_1682)

    def index_to_movie(x):     
        movie_list = [movie_id_name[i] for i in x]
        movie_str = '::'.join(movie_list)
        return movie_str
    

    train_data['movie_seq'] = train_data['seq_unpad'].apply(index_to_movie)

    return train_data

def data4frame_steam():
    game_id_name = dict()

    with open(r'./data/steam/id2name.txt', 'r') as f:
        for l in f.readlines():
            ll = l.strip('\n').split('::')
            game_name = str(ll[1])

            game_id_name[int(ll[0])] = game_name

    train_data = pd.read_pickle('./data/steam/train_data.df')
    train_data = train_data[train_data['len_seq'] >= 3]


    train_data['seq_unpad'] = train_data['seq'].apply(remove_3581)

    def index_to_movie(x):     
        movie_list = [game_id_name[i] for i in x]
        movie_str = '::'.join(movie_list)
        return movie_str

    train_data['movie_seq'] = train_data['seq_unpad'].apply(index_to_movie)

    return train_data

def data4frame_steam_test():
    game_id_name = dict()

    with open(r'./data/steam/id2name.txt', 'r') as f:
        for l in f.readlines():
            ll = l.strip('\n').split('::')
            game_name = str(ll[1])

            game_id_name[int(ll[0])] = game_name


    train_data = pd.read_pickle('./data/steam/Test_data.df')
    train_data = train_data[train_data['len_seq'] >= 3]


    train_data['seq_unpad'] = train_data['seq'].apply(remove_3581)

    def index_to_movie(x):     
        movie_list = [game_id_name[i] for i in x]
        movie_str = '::'.join(movie_list)
        return movie_str
    

    train_data['movie_seq'] = train_data['seq_unpad'].apply(index_to_movie)

    return train_data


def data4frame_game():
    game_id_name = dict()

    with open(r'./data/game_my/id2name.txt', 'r') as f:
        for l in f.readlines():
            ll = l.strip('\n').split('::')
            game_name = str(ll[1])

            game_id_name[int(ll[0])] = game_name

    train_data = pd.read_pickle('./data/game_my/train_data.df')
    train_data = train_data[train_data['len_seq'] >= 3]


    train_data['seq_unpad'] = train_data['seq']

    # def index_to_movie(x):     
    #     movie_list = [movie_id_name[i] for i in x]
    #     movie_str = ', '.join(movie_list) 
    #     return 'This user has watched ' + movie_str + ' in the previous.'

    def index_to_movie(x):     
        movie_list = [game_id_name[i] for i in x]
        movie_str = '::'.join(movie_list)
        return movie_str

    # def prefix_prompt(x):
    #     return 'A person has watched {} movies. These {} movies can be represented as: '.format(x, x)

    train_data['movie_seq'] = train_data['seq_unpad'].apply(index_to_movie)

    return train_data

def data4frame_game_test():
    game_id_name = dict()

    with open(r'./data/game_my/id2name.txt', 'r') as f:
        for l in f.readlines():
            ll = l.strip('\n').split('::')
            game_name = str(ll[1])

            game_id_name[int(ll[0])] = game_name

    train_data = pd.read_pickle('./data/game_my/Test_data.df')
    train_data = train_data[train_data['len_seq'] >= 3]


    train_data['seq_unpad'] = train_data['seq']

    # def index_to_movie(x):     
    #     movie_list = [movie_id_name[i] for i in x]
    #     movie_str = ', '.join(movie_list) 
    #     return 'This user has watched ' + movie_str + ' in the previous.'

    def index_to_movie(x):     
        movie_list = [game_id_name[i] for i in x]
        movie_str = '::'.join(movie_list)
        return movie_str

    # def prefix_prompt(x):
    #     return 'A person has watched {} movies. These {} movies can be represented as: '.format(x, x)

    train_data['movie_seq'] = train_data['seq_unpad'].apply(index_to_movie)

    return train_data

def data4frame_game2():
    game_id_name = dict()

    with open(r'./data/game_my2/id2name.txt', 'r') as f:
        for l in f.readlines():
            ll = l.strip('\n').split('::')
            game_name = str(ll[1])
            game_name = game_name.split(' - ')
            game_id_name[int(ll[0])] = game_name[0]

    train_data = pd.read_pickle('./data/game_my2/train_data.df')
    train_data = train_data[train_data['len_seq'] >= 3]


    train_data['seq_unpad'] = train_data['seq']

    # def index_to_movie(x):     
    #     movie_list = [movie_id_name[i] for i in x]
    #     movie_str = ', '.join(movie_list) 
    #     return 'This user has watched ' + movie_str + ' in the previous.'

    def index_to_movie(x):     
        movie_list = [game_id_name[i] for i in x]
        movie_str = '::'.join(movie_list)
        return movie_str

    # def prefix_prompt(x):
    #     return 'A person has watched {} movies. These {} movies can be represented as: '.format(x, x)

    train_data['movie_seq'] = train_data['seq_unpad'].apply(index_to_movie)

    return train_data

def data4frame_game2_test():
    game_id_name = dict()

    with open(r'./data/game_my2/id2name.txt', 'r') as f:
        for l in f.readlines():
            ll = l.strip('\n').split('::')
            game_name = str(ll[1])
            game_name = game_name.split(' - ')
            game_id_name[int(ll[0])] = game_name[0]

    train_data = pd.read_pickle('./data/game_my2/Test_data.df')
    train_data = train_data[train_data['len_seq'] >= 3]


    train_data['seq_unpad'] = train_data['seq']

    # def index_to_movie(x):     
    #     movie_list = [movie_id_name[i] for i in x]
    #     movie_str = ', '.join(movie_list) 
    #     return 'This user has watched ' + movie_str + ' in the previous.'

    def index_to_movie(x):     
        movie_list = [game_id_name[i] for i in x]
        movie_str = '::'.join(movie_list)
        return movie_str

    # def prefix_prompt(x):
    #     return 'A person has watched {} movies. These {} movies can be represented as: '.format(x, x)

    train_data['movie_seq'] = train_data['seq_unpad'].apply(index_to_movie)

    return train_data

class TrainingData(Dataset):
    def __init__(self, train_data, device):
        self.train_data = train_data 
        self.device=device
        super().__init__() 

    def __len__(self):
        return len(self.train_data['len_seq'])


    def __getitem__(self, i):
        temp = self.train_data.iloc[i]
        seq = torch.tensor(temp['seq']).to(self.device)
        len_seq = torch.tensor(temp['len_seq']).to(self.device)
        len_seq_list = temp['len_seq']
        sample = {'seq': seq, 
                  'len_seq': len_seq, 
                  'movie_seq': temp['movie_seq'],
                  'len_seq_list': len_seq_list
                  }

        return sample

class TrainingData_remove(Dataset):
    def __init__(self, train_data, l_rand, device):
        self.train_data = train_data 
        self.l_rand = l_rand
        self.device=device
        super().__init__() 

    def __len__(self):
        return len(self.train_data['len_seq'])


    def __getitem__(self, i):
        temp = self.train_data.iloc[i]

        seq = temp['seq']
        len_seq = temp['len_seq']
        movie_seq = temp['movie_seq']
        if len_seq-self.l_rand >= 0:
            remove_idx = random.randint(len_seq-self.l_rand, len_seq-1)
        else:
            remove_idx = random.randint(0, len_seq-1)
        # print(remove_idx, seq[remove_idx], movie_seq)
        id_remove = seq[remove_idx]
        movie_remove = movie_seq.split('::')[remove_idx]
        seq_remove = seq[:remove_idx] + seq[remove_idx+1:] + [1682]


        seq = torch.tensor(temp['seq']).to(self.device)
        len_seq = torch.tensor(temp['len_seq']).to(self.device)
        seq_remove = torch.tensor(seq_remove).to(self.device)
        id_remove = torch.tensor(id_remove).to(self.device)

        len_seq_list = temp['len_seq']

        sample = {'seq': seq, 
        'len_seq': len_seq, 
        'movie_seq': movie_seq, 
        'seq_remove': seq_remove, 
        'id_remove': id_remove, 
        'movie_remove': movie_remove,
        'len_seq_list': len_seq_list}

        return sample



if __name__ == "__main__":

    training_data = TrainingData_remove(data4frame(), 10, device='cpu')


    train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)

    for i, sample in enumerate(train_dataloader):

        seq = sample["seq"]
        len_seq = sample['len_seq']
        seq_remove = sample['seq_remove']
        id_remove= sample['id_remove']
        movie_remove = sample['movie_remove']
        name_list = sample['movie_seq']
        len_seq_list = sample['len_seq_list']

        # seq2 = seq.scatter(1, len_seq.view(len_seq.shape[0], 1) - 1, 1682)

        # len_seq2 = len_seq - 1

        print(seq)
        print(len_seq)
        print(seq_remove)
        print(id_remove)
        print(movie_remove)
        print(name_list)
        print(len_seq_list)

    