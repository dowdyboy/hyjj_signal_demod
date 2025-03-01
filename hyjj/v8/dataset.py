import math
import pandas as pd
from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas
import random


def process_nco_signal_infer(signal_I_real, signal_Q_imag):
    signal_IQ = signal_I_real + 1j * signal_Q_imag
    Fs = 20e6  # 采样率20MHz
    N = len(signal_IQ)  # IQ数据长度

    signal = np.zeros(N, dtype=np.complex64)
    signal[:N] = signal_IQ[:N]
    t = (np.arange(N) / Fs)

    drop = 10  # 边缘偏10%
    # 计算频谱
    spectrum = np.fft.fftshift(np.fft.fft(signal))  # FFT
    freq = np.arange(-Fs/2, Fs/2, Fs/N) + (Fs / N)   # 频率对应的坐标
    power_spectrum = np.abs(spectrum) ** 2  # 计算功率谱
    # 累计归一化功率谱
    cumulative_power = np.cumsum(power_spectrum) / np.sum(power_spectrum)
    # 找到 5% 和 95% 的频率范围
    low_idx = np.where(cumulative_power >= drop / 100)[0][0]  # 累计功率达到 5%
    high_idx = np.where(cumulative_power >= (100 - drop) / 100)[0][0]  # 累计功率达到 95%
    f_low = freq[low_idx]  # 下边界频率
    f_high = freq[high_idx]  # 上边界频率

    freq_mid = (f_low + f_high) / 2
    bandwidth = f_high - f_low

    # fir 低通
    Fpass = bandwidth / 2 * 1.1
    Fstop = Fpass * 1.2
    # hlpf = get_lfp(Fs, Fpass, Fstop, False)
    # 频域预处理
    F_NCO = freq_mid / 1e6 - 0.0
    noc = np.exp(-1j * 2 * np.pi * (F_NCO * 1e6) * t)
    snf = signal * noc

    F_NCO_2 = freq_mid / 1e6 - 0.2
    noc_2 = np.exp(-1j * 2 * np.pi * (F_NCO_2 * 1e6) * t)
    snf_2 = signal * noc_2

    F_NCO_4 = freq_mid / 1e6 - 0.4
    noc_4 = np.exp(-1j * 2 * np.pi * (F_NCO_4 * 1e6) * t)
    snf_4 = signal * noc_4

    F_NCO_2p = freq_mid / 1e6 + 0.2
    noc_2p = np.exp(-1j * 2 * np.pi * (F_NCO_2p * 1e6) * t)
    snf_2p = signal * noc_2p

    F_NCO_4p = freq_mid / 1e6 + 0.4
    noc_4p = np.exp(-1j * 2 * np.pi * (F_NCO_4p * 1e6) * t)
    snf_4p = signal * noc_4p

    return (np.real(snf).astype(np.float32), np.imag(snf).astype(np.float32),
            np.real(snf_2).astype(np.float32), np.imag(snf_2).astype(np.float32),
            np.real(snf_4).astype(np.float32), np.imag(snf_4).astype(np.float32),
            np.real(snf_2p).astype(np.float32), np.imag(snf_2p).astype(np.float32),
            np.real(snf_4p).astype(np.float32), np.imag(snf_4p).astype(np.float32),
            np.abs(spectrum))


def process_nco_signal(signal_I_real, signal_Q_imag, code_num, code, code_width):
    # 在Python中，使用numpy的complex函数来创建复数数组
    signal_IQ = signal_I_real + 1j * signal_Q_imag

    # 参数读取
    Fs = 20e6  # 采样率20MHz
    N = len(signal_IQ)  # IQ数据长度
    # t = (np.arange(N) / Fs)  # 时间序列
    # f = np.arange(-Fs/2, Fs/2, Fs/N) + (Fs / N)    # 频率序列

    # code_num = np.sum(~np.isnan( np.array(df.iloc[:, 2]) ))
    # code = np.array(df.iloc[:code_num, 2], np.int32)
    # code_width = float(df.iloc[0, 4] / 1e6)  # 码元宽度

    code_width = float(code_width / 1e6)

    smp_per_code = int(round(code_width * Fs))
    sample_num = smp_per_code * code_num

    if N > sample_num:
        signal = np.zeros(sample_num, dtype=np.complex64)
        signal[:sample_num] = signal_IQ[:sample_num]
        t = (np.arange(sample_num) / Fs)
    else:
        signal = np.zeros(N, dtype=np.complex64)
        signal[:N] = signal_IQ[:N]
        t = (np.arange(N) / Fs)

    ####  尝试寻找频谱中心

    drop = 10  # 边缘偏10%
    # 计算频谱
    spectrum = np.fft.fftshift(np.fft.fft(signal))  # FFT
    freq = np.arange(-Fs/2, Fs/2, Fs/N) + (Fs / N)   # 频率对应的坐标
    power_spectrum = np.abs(spectrum) ** 2  # 计算功率谱
    # 累计归一化功率谱
    cumulative_power = np.cumsum(power_spectrum) / np.sum(power_spectrum)
    # 找到 5% 和 95% 的频率范围
    low_idx = np.where(cumulative_power >= drop / 100)[0][0]  # 累计功率达到 5%
    high_idx = np.where(cumulative_power >= (100 - drop) / 100)[0][0]  # 累计功率达到 95%
    f_low = freq[low_idx]  # 下边界频率
    f_high = freq[high_idx]  # 上边界频率


    freq_mid = (f_low + f_high) / 2
    bandwidth = f_high - f_low

    ####### 分析
    ####### 成型滤波器 升余弦 根升余弦 高斯 fir低通
    ####### 升余弦系列
    span = 10
    sps = smp_per_code
    # 成型滤波器
    # hc = rcosdesign(0.25, span, sps)  # 升余弦
    # hrc = scipy.signal.rcosine(0.25, span, sps, True)  # 根升余弦
    # hg = scipy.signal.gaussian(span * sps, std=2e6 * span * sps / Fs)  # 高斯
    # fir 低通
    Fpass = bandwidth / 2 * 1.1
    Fstop = Fpass * 1.2
    # hlpf = get_lfp(Fs, Fpass, Fstop, False)
    # 频域预处理
    F_NCO = freq_mid / 1e6 - 0.0
    noc = np.exp(-1j * 2 * np.pi * (F_NCO * 1e6) * t)
    snf = signal * noc

    F_NCO_2 = freq_mid / 1e6 - 0.2
    noc_2 = np.exp(-1j * 2 * np.pi * (F_NCO_2 * 1e6) * t)
    snf_2 = signal * noc_2

    F_NCO_4 = freq_mid / 1e6 - 0.4
    noc_4 = np.exp(-1j * 2 * np.pi * (F_NCO_4 * 1e6) * t)
    snf_4 = signal * noc_4

    F_NCO_2p = freq_mid / 1e6 + 0.2
    noc_2p = np.exp(-1j * 2 * np.pi * (F_NCO_2p * 1e6) * t)
    snf_2p = signal * noc_2p

    F_NCO_4p = freq_mid / 1e6 + 0.4
    noc_4p = np.exp(-1j * 2 * np.pi * (F_NCO_4p * 1e6) * t)
    snf_4p = signal * noc_4p

    return (np.real(snf).astype(np.float32), np.imag(snf).astype(np.float32),
            np.real(snf_2).astype(np.float32), np.imag(snf_2).astype(np.float32),
            np.real(snf_4).astype(np.float32), np.imag(snf_4).astype(np.float32),
            np.real(snf_2p).astype(np.float32), np.imag(snf_2p).astype(np.float32),
            np.real(snf_4p).astype(np.float32), np.imag(snf_4p).astype(np.float32),
            np.abs(spectrum))


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
                continue
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


def gen_amr_data_aug(i_data, q_data,
                     i_data_2, q_data_2,
                     i_data_4, q_data_4,
                     i_data_2p, q_data_2p,
                     i_data_4p, q_data_4p,
                     sample_len, sample_type='center', flip_rate=0.0):
    data_len = len(i_data)
    if data_len < sample_len:
        i_data = np.tile(i_data, math.ceil(sample_len / data_len))
        q_data = np.tile(q_data, math.ceil(sample_len / data_len))
        i_data = i_data[0:sample_len]
        q_data = q_data[0:sample_len]
        i_data_2 = np.tile(i_data_2, math.ceil(sample_len / data_len))
        q_data_2 = np.tile(q_data_2, math.ceil(sample_len / data_len))
        i_data_2 = i_data_2[0:sample_len]
        q_data_2 = q_data_2[0:sample_len]
        i_data_4 = np.tile(i_data_4, math.ceil(sample_len / data_len))
        q_data_4 = np.tile(q_data_4, math.ceil(sample_len / data_len))
        i_data_4 = i_data_4[0:sample_len]
        q_data_4 = q_data_4[0:sample_len]
        i_data_2p = np.tile(i_data_2p, math.ceil(sample_len / data_len))
        q_data_2p = np.tile(q_data_2p, math.ceil(sample_len / data_len))
        i_data_2p = i_data_2p[0:sample_len]
        q_data_2p = q_data_2p[0:sample_len]
        i_data_4p = np.tile(i_data_4p, math.ceil(sample_len / data_len))
        q_data_4p = np.tile(q_data_4p, math.ceil(sample_len / data_len))
        i_data_4p = i_data_4p[0:sample_len]
        q_data_4p = q_data_4p[0:sample_len]
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
        i_data_2 = i_data_2[start_idx:start_idx + sample_len]
        q_data_2 = q_data_2[start_idx:start_idx + sample_len]
        i_data_4 = i_data_4[start_idx:start_idx + sample_len]
        q_data_4 = q_data_4[start_idx:start_idx + sample_len]
        i_data_2p = i_data_2p[start_idx:start_idx + sample_len]
        q_data_2p = q_data_2p[start_idx:start_idx + sample_len]
        i_data_4p = i_data_4p[start_idx:start_idx + sample_len]
        q_data_4p = q_data_4p[start_idx:start_idx + sample_len]

    if random.random() < flip_rate:
        i_data = i_data[::-1]
        q_data = q_data[::-1]
        i_data_2 = i_data_2[::-1]
        q_data_2 = q_data_2[::-1]
        i_data_4 = i_data_4[::-1]
        q_data_4 = q_data_4[::-1]
        i_data_2p = i_data_2p[::-1]
        q_data_2p = q_data_2p[::-1]
        i_data_4p = i_data_4p[::-1]
        q_data_4p = q_data_4p[::-1]

    return (i_data, q_data,
            i_data_2, q_data_2,
            i_data_4, q_data_4,
            i_data_2p, q_data_2p,
            i_data_4p, q_data_4p)


class HyjjAMRCWDataset(Dataset):

    def __init__(self, dataset, sample_len, sample_type='center', flip_rate=0.0):
        super().__init__()
        self.dataset = dataset
        self.sample_len = sample_len
        self.sample_type = sample_type
        self.flip_rate = flip_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        i_data, q_data, code_series, amr_label, code_width = self.dataset[idx]
        (i_data, q_data,
         i_data_2, q_data_2,
         i_data_4, q_data_4,
         i_data_2p, q_data_2p,
         i_data_4p, q_data_4p,
         spectrum) = process_nco_signal(i_data, q_data, code_series.shape[0], code_series, code_width)
        (i_data, q_data,
         i_data_2, q_data_2,
         i_data_4, q_data_4,
         i_data_2p, q_data_2p,
         i_data_4p, q_data_4p) = gen_amr_data_aug(i_data, q_data,
                                          i_data_2, q_data_2,
                                          i_data_4, q_data_4,
                                          i_data_2p, q_data_2p,
                                          i_data_4p, q_data_4p,
                                          self.sample_len, self.sample_type, self.flip_rate)
        return np.array([
            i_data, q_data,
            i_data_2, q_data_2,
            i_data_4, q_data_4,
            i_data_2p, q_data_2p,
            i_data_4p, q_data_4p,
        ]), amr_label, code_width


def gen_demod_data_aug(i_data, q_data,
                       i_data_2, q_data_2,
                       i_data_4, q_data_4,
                       i_data_2p, q_data_2p,
                       i_data_4p, q_data_4p,
                       sample_len, code_series, code_width, sample_type='rand', ):
    data_len = len(i_data)
    Fs = 20e6  # 采样率20MHz
    code_width = float(code_width / 1e6)
    smp_per_code = int(round(code_width * Fs))
    cw_sample_len = smp_per_code * len(code_series)
    cs_data = np.array(code_series, dtype=np.int64)
    cs_data = np.repeat(cs_data, smp_per_code)
    pos_data = np.arange(1, len(code_series)+1)
    pos_data = pos_data.astype(np.float32)
    pos_data = pos_data / np.max(pos_data)
    pos_data = np.repeat(pos_data, smp_per_code)

    if cw_sample_len < data_len:
        new_cs_data = np.zeros(i_data.shape, dtype=np.int64)
        new_cs_data[:cw_sample_len] = cs_data
        cs_data = new_cs_data
        new_pos_data = np.zeros(i_data.shape, dtype=np.float32)
        new_pos_data[:cw_sample_len] = pos_data
        pos_data = new_pos_data
    elif cw_sample_len > data_len:
        cs_data = cs_data[:data_len]
        pos_data = pos_data[:data_len]

    if data_len < sample_len:
        i_data = np.tile(i_data, math.ceil(sample_len / data_len))
        q_data = np.tile(q_data, math.ceil(sample_len / data_len))
        i_data_2 = np.tile(i_data_2, math.ceil(sample_len / data_len))
        q_data_2 = np.tile(q_data_2, math.ceil(sample_len / data_len))
        i_data_4 = np.tile(i_data_4, math.ceil(sample_len / data_len))
        q_data_4 = np.tile(q_data_4, math.ceil(sample_len / data_len))
        i_data_2p = np.tile(i_data_2p, math.ceil(sample_len / data_len))
        q_data_2p = np.tile(q_data_2p, math.ceil(sample_len / data_len))
        i_data_4p = np.tile(i_data_4p, math.ceil(sample_len / data_len))
        q_data_4p = np.tile(q_data_4p, math.ceil(sample_len / data_len))

        cs_data = np.tile(cs_data, math.ceil(sample_len / data_len))
        pos_data = np.tile(pos_data, math.ceil(sample_len / data_len))
        i_data = i_data[0:sample_len]
        q_data = q_data[0:sample_len]
        i_data_2 = i_data_2[0:sample_len]
        q_data_2 = q_data_2[0:sample_len]
        i_data_4 = i_data_4[0:sample_len]
        q_data_4 = q_data_4[0:sample_len]
        i_data_2p = i_data_2p[0:sample_len]
        q_data_2p = q_data_2p[0:sample_len]
        i_data_4p = i_data_4p[0:sample_len]
        q_data_4p = q_data_4p[0:sample_len]

        cs_data = cs_data[0:sample_len]
        pos_data = pos_data[0:sample_len]
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
        i_data_2 = i_data_2[start_idx:start_idx + sample_len]
        q_data_2 = q_data_2[start_idx:start_idx + sample_len]
        i_data_4 = i_data_4[start_idx:start_idx + sample_len]
        q_data_4 = q_data_4[start_idx:start_idx + sample_len]
        i_data_2p = i_data_2p[start_idx:start_idx + sample_len]
        q_data_2p = q_data_2p[start_idx:start_idx + sample_len]
        i_data_4p = i_data_4p[start_idx:start_idx + sample_len]
        q_data_4p = q_data_4p[start_idx:start_idx + sample_len]

        cs_data = cs_data[start_idx:start_idx + sample_len]
        pos_data = pos_data[start_idx:start_idx + sample_len]

    return (i_data, q_data,
            i_data_2, q_data_2,
            i_data_4, q_data_4,
            i_data_2p, q_data_2p,
            i_data_4p, q_data_4p,
            pos_data, cs_data)


def gen_demod_data_aug_infer(i_data, q_data,
                             i_data_2, q_data_2,
                             i_data_4, q_data_4,
                             i_data_2p, q_data_2p,
                             i_data_4p, q_data_4p,
                             sample_len, code_width, ):
    data_len = len(i_data)
    Fs = 20e6  # 采样率20MHz
    code_width = float(code_width / 1e6)
    smp_per_code = int(round(code_width * Fs))
    pos_data = np.arange(1, math.ceil(data_len // smp_per_code) + 1)
    pos_data = pos_data.astype(np.float32)
    pos_data = pos_data / np.max(pos_data)
    pos_data = np.repeat(pos_data, smp_per_code)
    if len(pos_data) < data_len:
        new_pos_data = np.zeros(i_data.shape, dtype=np.float32)
        new_pos_data[:len(pos_data)] = pos_data
        new_pos_data[len(pos_data):] = pos_data[len(pos_data) - (data_len - len(pos_data)):]
        pos_data = new_pos_data
    elif len(pos_data) > data_len:
        pos_data = pos_data[:data_len]

    ret_i, ret_q, ret_pos = [], [], []
    if data_len < sample_len:
        i_data = np.tile(i_data, math.ceil(sample_len / data_len))
        q_data = np.tile(q_data, math.ceil(sample_len / data_len))
        i_data_2 = np.tile(i_data_2, math.ceil(sample_len / data_len))
        q_data_2 = np.tile(q_data_2, math.ceil(sample_len / data_len))
        i_data_4 = np.tile(i_data_4, math.ceil(sample_len / data_len))
        q_data_4 = np.tile(q_data_4, math.ceil(sample_len / data_len))
        i_data_2p = np.tile(i_data_2p, math.ceil(sample_len / data_len))
        q_data_2p = np.tile(q_data_2p, math.ceil(sample_len / data_len))
        i_data_4p = np.tile(i_data_4p, math.ceil(sample_len / data_len))
        q_data_4p = np.tile(q_data_4p, math.ceil(sample_len / data_len))

        pos_data = np.tile(pos_data, math.ceil(sample_len / data_len))
        i_data = i_data[0:sample_len]
        q_data = q_data[0:sample_len]
        i_data_2 = i_data_2[0:sample_len]
        q_data_2 = q_data_2[0:sample_len]
        i_data_4 = i_data_4[0:sample_len]
        q_data_4 = q_data_4[0:sample_len]
        i_data_2p = i_data_2p[0:sample_len]
        q_data_2p = q_data_2p[0:sample_len]
        i_data_4p = i_data_4p[0:sample_len]
        q_data_4p = q_data_4p[0:sample_len]

        pos_data = pos_data[0:sample_len]
        ret_i.append(i_data)
        ret_q.append(q_data)
        ret_pos.append(pos_data)
    elif data_len > sample_len:
        cnt = math.ceil(data_len / sample_len)
        for k in range(cnt):
            if k == cnt - 1:
                tmp_data_len = data_len - k*sample_len
                tmp_i = i_data[k*sample_len:data_len]
                tmp_q = q_data[k*sample_len:data_len]
                tmp_pos = pos_data[k*sample_len:data_len]
                if tmp_data_len < sample_len:
                    tmp_i = np.tile(tmp_i, math.ceil(sample_len / tmp_data_len))
                    tmp_q = np.tile(tmp_q, math.ceil(sample_len / tmp_data_len))
                    tmp_pos = np.tile(tmp_pos, math.ceil(sample_len / tmp_data_len))
                    tmp_i = tmp_i[0:sample_len]
                    tmp_q = tmp_q[0:sample_len]
                    tmp_pos = tmp_pos[0:sample_len]
                    ret_i.append(tmp_i)
                    ret_q.append(tmp_q)
                    ret_pos.append(tmp_pos)
            else:
                tmp_i = i_data[k*sample_len:k*sample_len+sample_len]
                tmp_q = q_data[k*sample_len:k*sample_len+sample_len]
                tmp_pos = pos_data[k*sample_len:k*sample_len+sample_len]
                ret_i.append(tmp_i)
                ret_q.append(tmp_q)
                ret_pos.append(tmp_pos)
    else:
        ret_i.append(i_data)
        ret_q.append(q_data)
        ret_pos.append(pos_data)

    return ret_i, ret_q, ret_pos


class HyjjDemodDataset(Dataset):

    def __init__(self, dataset, data_len, position_code=False):
        super().__init__()
        self.dataset = dataset
        self.data_len = data_len
        self.position_code = position_code

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        i_data, q_data, code_series, amr_label, code_width = self.dataset[idx]
        (i_data, q_data,
         i_data_2, q_data_2,
         i_data_4, q_data_4,
         i_data_2p, q_data_2p,
         i_data_4p, q_data_4p,
         spectrum) = process_nco_signal(i_data, q_data, code_series.shape[0], code_series, code_width)
        (i_data, q_data,
         i_data_2, q_data_2,
         i_data_4, q_data_4,
         i_data_2p, q_data_2p,
         i_data_4p, q_data_4p,
         pos_data, cs_data) = gen_demod_data_aug(i_data, q_data,
                                                               i_data_2, q_data_2,
                                                               i_data_4, q_data_4,
                                                               i_data_2p, q_data_2p,
                                                               i_data_4p, q_data_4p,
                                                               self.data_len, code_series, code_width)
        # extend_i_data = np.interp(
        #     np.linspace(0, len(i_data)-1, self.data_len),
        #     np.linspace(0, len(i_data)-1, len(i_data)),
        #     i_data)
        # extend_q_data = np.interp(
        #     np.linspace(0, len(q_data)-1, self.data_len),
        #     np.linspace(0, len(q_data)-1, len(q_data)),
        #     q_data
        # )
        # extend_code_series = np.zeros(extend_q_data.shape, dtype=np.int64)
        if self.position_code:
            return np.array([
                i_data, q_data,
                i_data_2, q_data_2,
                i_data_4, q_data_4,
                i_data_2p, q_data_2p,
                i_data_4p, q_data_4p,
                pos_data], dtype=np.float32), cs_data
        else:
            return np.array([
                i_data, q_data,
                i_data_2, q_data_2,
                i_data_4, q_data_4,
                i_data_2p, q_data_2p,
                i_data_4p, q_data_4p
            ], dtype=np.float32), cs_data


if __name__ == '__main__':
    mod_names = [
        'BPSK', 'QPSK', '8PSK', 'MSK', '8QAM',
        '16QAM', '32QAM', '8APSK', '16APSK', '32APSK'
    ]
    train_base_ds = HyjjDataset(
        root_dir=r'E:\BaiduNetdiskDownload\hyjj_signal_demod\train_data',
        train_ratio=0.9,
        is_train=True,
        # label_filter=['BPSK'],
    )

    # ds = HyjjAMRCWDataset(train_base_ds, 2000, flip_rate=0.5)
    # ds[0]

    ds = HyjjDemodDataset(train_base_ds, 2000, True)
    for i in range(len(ds)):
        x, y = ds[i]
        print()


