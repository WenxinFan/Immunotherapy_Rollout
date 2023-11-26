import numpy as np
import h5py
import torch
from torch.utils.data import Dataset


class Our_Dataset(Dataset):
    def __init__(self, data_path, data_type='train'):
        super(Our_Dataset, self).__init__()

        self.data_path = data_path

        if data_type == 'train':
            self.train = True
            self.val = False
            self.test = False
            self.x = np.load(data_path + 'train_x_states.npy', allow_pickle=True)
            self.acts = np.load(data_path + 'train_acts.npy')
            self.info = np.load(data_path + 'train_info.npy')
            self.y_all = np.load(data_path + 'train_y.npy', allow_pickle=True)
            # self.y_states = np.load(data_path + 'train_y_states.npy', allow_pickle=True)
            # self.x = np.array(h5py.File(data_path + 'train_x_states.npy', 'r')['train_x_states'])
            # self.acts = np.array(h5py.File(data_path + 'train_acts.npy', 'r')['train_acts'])
            # self.y_all = np.array(h5py.File(data_path + 'train_y_states.npy', 'r')['train_y_states'])
            self.shuffle = True

        elif data_type == 'val':
            self.train = False
            self.val = True
            self.test = False
            self.x = np.load(data_path + 'valid_x_states.npy')
            self.acts = np.load(data_path + 'valid_acts.npy')
            # self.y_states = np.load(data_path + 'valid_y_states.npy')
            self.shuffle = False

        elif data_type == 'test':
            self.train = False
            self.val = False
            self.test = True
            self.x = np.load(data_path + 'test_x_states.npy', allow_pickle=True)
            self.acts = np.load(data_path + 'test_acts.npy')
            self.info = np.load(data_path + 'test_info.npy')
            self.y_all = np.load(data_path + 'test_y.npy', allow_pickle=True)
            # self.y_states = np.load(data_path + 'test_y_states.npy', allow_pickle=True)
            self.shuffle = False

        self.x = self.x.astype(float)
        self.y_all = self.y_all.astype(float)
        # self.y_states = self.y_states.astype(float)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x_states = torch.FloatTensor(self.x[index, :, :])
        y_all = torch.FloatTensor(self.y_all[index, :, :])
        # y_states = torch.FloatTensor(self.y_states[index, :, :])
        acts = torch.FloatTensor(self.acts[index, :])
        info = torch.FloatTensor(self.info[index, :])
        return x_states, acts, info, y_all
