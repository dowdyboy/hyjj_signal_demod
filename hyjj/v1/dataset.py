from torch.utils.data import Dataset
import os
import numpy as np
import pandas
import random


class HyjjDataset(Dataset):

    def __init__(self, root_dir, train_ratio=0.9, is_train=True, label_filter=None, rand_seed=114514):
        super().__init__()
        random.seed(rand_seed)
        self.root_dir = root_dir
        self.label_filter = label_filter
        self.label_map = {
            'BPSK': 1,
            'QPSK': 2,
            '8PSK': 3,
            'MSK': 4,
            '8QAM': 5,
            '16QAM': 6,
            '32QAM': 7,
            '8APSK': 8,
            '16APSK': 9,
            '32APSK': 10
        }
        self.file_path_list = []
        for sig_type_name in os.listdir(root_dir):
            if sig_type_name not in self.label_map.keys():
                print(f'error signal type: {sig_type_name}')
                exit(1)
            if label_filter is None or sig_type_name in label_filter:
                sig_type_path = os.path.join(root_dir, sig_type_name)
                filename_list = list(sorted(os.listdir(sig_type_path)))
                random.shuffle(filename_list)
                if is_train:
                    filename_list = filename_list[:int(len(filename_list)*train_ratio)]
                else:
                    filename_list = filename_list[int(len(filename_list)*train_ratio):]
                filename_list = list(map(lambda x: os.path.join(sig_type_path, x), filename_list))
                self.file_path_list.extend(filename_list)

    def __len__(self):
        return len(self.file_path_list)

    def __getitem__(self, idx):
        filepath = self.file_path_list[idx]
        csv = pandas.read_csv(filepath, header=None, )
        i_data = np.array(list(csv[0]), dtype=np.float32)
        q_data = np.array(list(csv[1]), dtype=np.float32)
        code_series = np.array(list(csv[2].dropna()), dtype=np.int32)
        amr_label = int(csv[3][0]) - 1
        code_width = float(csv[4][0])
        return i_data, q_data, code_series, amr_label, code_width


def gen_amr_data(i_data, q_data, sample_len, sample_type='center', padding_type='center'):
    assert sample_type in ['center', 'rand']
    assert padding_type in ['center', 'right']
    data_len = len(i_data)
    if data_len < sample_len:
        delta_len = sample_len - data_len
        if padding_type == 'center':
            if delta_len % 2 == 0:
                l_pad_size = delta_len // 2
                r_pad_size = l_pad_size
            else:
                l_pad_size = delta_len // 2
                r_pad_size = l_pad_size + 1
            i_data = np.concatenate(
                [np.zeros([l_pad_size, ], dtype=np.float32),
                 i_data,
                 np.zeros([r_pad_size, ], dtype=np.float32)],
                axis=0
            )
            q_data = np.concatenate(
                [np.zeros([l_pad_size, ], dtype=np.float32),
                 q_data,
                 np.zeros([r_pad_size, ], dtype=np.float32)],
                axis=0
            )
        elif padding_type == 'right':
            i_data = np.concatenate(
                [i_data, np.zeros([delta_len, ], dtype=np.float32)], axis=0,
            )
            q_data = np.concatenate(
                [q_data, np.zeros([delta_len, ], dtype=np.float32)], axis=0
            )
    else:
        delta_len = data_len - sample_len
        if sample_type == 'center':
            start_idx = delta_len // 2
        elif sample_type == 'rand':
            start_idx = random.randint(0, delta_len // 2)
        else:
            start_idx = 0
        i_data = i_data[start_idx:start_idx + sample_len]
        q_data = q_data[start_idx:start_idx + sample_len]

    return i_data, q_data


def gen_series_data(i_data, q_data, code_series, code_width):
    pass


class HyjjAMRDataset(Dataset):

    def __init__(self, dataset, sample_len, sample_type='center', padding_type='center'):
        super().__init__()
        self.dataset = dataset
        self.sample_len = sample_len
        self.sample_type = sample_type
        self.padding_type = padding_type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        i_data, q_data, _, amr_label, _ = self.dataset[idx]
        i_data, q_data = gen_amr_data(i_data, q_data, self.sample_len, self.sample_type, self.padding_type)
        return np.array([i_data, q_data]), amr_label


class HyjjAMRCWDataset(Dataset):

    def __init__(self, dataset, sample_len, sample_type='center', padding_type='center'):
        super().__init__()
        self.dataset = dataset
        self.sample_len = sample_len
        self.sample_type = sample_type
        self.padding_type = padding_type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        i_data, q_data, _, amr_label, code_width = self.dataset[idx]
        i_data, q_data = gen_amr_data(i_data, q_data, self.sample_len, self.sample_type, self.padding_type)
        return np.array([i_data, q_data]), amr_label, code_width


if __name__ == '__main__':
    # ds = HyjjDataset(r'E:\BaiduNetdiskDownload\train_data', is_train=True)
    # ds2 = HyjjDataset(r'E:\BaiduNetdiskDownload\train_data', is_train=False, label_filter=['BPSK'])
    # i_data, q_data, code_series, amr_label, code_width = ds[1]
    i_data = np.ones([129, ], dtype=np.float32)
    q_data = np.ones([129, ], dtype=np.float32)
    new_i_data, new_q_data = gen_amr_data(i_data, q_data, 128, sample_type='rand', padding_type='right')
    print()
