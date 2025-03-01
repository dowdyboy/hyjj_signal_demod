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

    return np.real(snf).astype(np.float32), np.imag(snf).astype(np.float32), np.abs(spectrum)


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

    return np.real(snf).astype(np.float32), np.imag(snf).astype(np.float32), np.abs(spectrum)


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


def gen_amr_data_aug(i_data, q_data, sample_len, sample_type='center', flip_rate=0.0):
    data_len = len(i_data)
    if data_len < sample_len:
        i_data = np.tile(i_data, math.ceil(sample_len / data_len))
        q_data = np.tile(q_data, math.ceil(sample_len / data_len))
        i_data = i_data[0:sample_len]
        q_data = q_data[0:sample_len]
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

    if random.random() < flip_rate:
        i_data = i_data[::-1]
        q_data = q_data[::-1]

    return i_data, q_data


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
        i_data, q_data, spectrum = process_nco_signal(i_data, q_data, code_series.shape[0], code_series, code_width)
        i_data, q_data = gen_amr_data_aug(i_data, q_data, self.sample_len, self.sample_type, self.flip_rate)
        return np.array([i_data, q_data]), amr_label, code_width


# class HyjjDemodDataset(Dataset):
#
#     def __init__(self, dataset, signal_length, ):
#         super().__init__()
#         self.num_class_map = {
#             'BPSK': 2,
#             'QPSK': 4,
#             '8PSK': 8,
#             'MSK': 2,
#             '8QAM': 8,
#             '16QAM': 16,
#             '32QAM': 32,
#             '8APSK': 8,
#             '16APSK': 16,
#             '32APSK': 32
#         }
#         self.dataset = dataset
#         self.signal_length = signal_length
#         self.num_class = self.num_class_map[self.dataset.label_filter[0]]
#         Fs = 20e6
#         for i in range(len(self.dataset)):
#             i_data, q_data, code_series, amr_label, code_width = self.dataset[i]
#             i_data, q_data, spectrum = process_nco_signal(i_data, q_data, code_series.shape[0], code_series, code_width)
#             smp_per_code = round(code_width / 1e6 * Fs)
#             if smp_per_code == signal_length:
#
#                 print()
#
#         #     if smp_per_code in count_map.keys():
#         #         count_map[smp_per_code] += 1
#         #     else:
#         #         count_map[smp_per_code] = 1
#         # print(self.dataset.label_filter, list(sorted(count_map.items(), key=lambda x:x[1], reverse=True)))
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         pass


class HyjjDemodDataset(Dataset):

    def __init__(self, root_dir, need_type_name, signal_len=256):
        super().__init__()
        self.root_dir = root_dir
        self.need_type_name = need_type_name
        self.signal_len = signal_len
        self.filepath_list = []
        for type_name in os.listdir(root_dir):
            if type_name == need_type_name:
                type_sub_dir = os.path.join(root_dir, type_name)
                for s_len in os.listdir(type_sub_dir):
                    s_len_sub_dir = os.path.join(type_sub_dir, s_len)
                    for filename in os.listdir(s_len_sub_dir):
                        filepath = os.path.join(s_len_sub_dir, filename)
                        self.filepath_list.append(filepath)
        assert len(self.filepath_list) != 0

    def __len__(self):
        return len(self.filepath_list)

    def __getitem__(self, idx):
        filepath = self.filepath_list[idx]
        csv = pandas.read_csv(filepath, header=None, )
        i_data = np.array(list(csv[0]), dtype=np.float32)
        q_data = np.array(list(csv[1]), dtype=np.float32)
        code = int(os.path.basename(filepath).split('.')[0].split('_')[1])
        extend_i_data = np.interp(
                  np.linspace(0, len(i_data)-1, self.signal_len),
                  np.linspace(0, len(i_data)-1, len(i_data)),
                  i_data)
        extend_q_data = np.interp(
            np.linspace(0, len(q_data)-1, self.signal_len),
            np.linspace(0, len(q_data)-1, len(q_data)),
            q_data
        )
        return np.array([extend_i_data, extend_q_data], dtype=np.float32), code


if __name__ == '__main__':
    # mod_names = [
    #     'BPSK', 'QPSK', '8PSK', 'MSK', '8QAM',
    #     '16QAM', '32QAM', '8APSK', '16APSK', '32APSK'
    # ]
    # train_base_ds = HyjjDataset(
    #     root_dir=r'E:\BaiduNetdiskDownload\hyjj_signal_demod\train_data',
    #     train_ratio=0.9,
    #     is_train=True,
    #     label_filter=['BPSK'],
    # )
    # ds = HyjjDemodDataset(train_base_ds, 18)

    ds = HyjjDemodDataset(r'E:\BaiduNetdiskDownload\hyjj_signal_demod\demod_data', )
    extend_i_data, extend_q_data, code = ds[0]
    print()


if __name__ == '__main__123':
    train_base_ds = HyjjDataset(
        root_dir=r'E:\BaiduNetdiskDownload\hyjj_signal_demod\train_data',
        train_ratio=0.9,
        is_train=True,
        rand_seed=0,
    )
    ds = HyjjAMRCWDataset(train_base_ds, 2000, flip_rate=1.0)
    ds[0]
    print()



if __name__ == '__main__123123':
    train_base_ds = HyjjDataset(
        root_dir=r'E:\BaiduNetdiskDownload\hyjj_signal_demod\train_data',
        train_ratio=0.9,
        is_train=True,
        rand_seed=0,
    )
    ds = HyjjAMRCWDataset(train_base_ds, 512, sample_type='rand', )

    for i in range(3):
        iq_data, amr_label, code_width = ds[i]
        new_i_data, new_q_data = iq_data[0, :], iq_data[1, :]
        # new_i_data, new_q_data = process_nco_signal(i_data, q_data, code_series.shape[0], code_series, code_width)

        t = np.arange(new_i_data.shape[0]) / 20e6
        plt.figure()
        plt.subplot(211)
        # plt.plot(t, i_data, 'r', t, q_data, 'b')
        plt.subplot(212)
        plt.plot(t, new_i_data, 'r', t, new_q_data, 'b')
        plt.show()

    print()


# QPSK  120  -0.04
# BPSK  120  -0.03

if __name__ == '__main__123':
    data_file_cnt = 120
    mode_fun = 'QPSK'
    file_name = os.path.join(r'E:\BaiduNetdiskDownload\hyjj_signal_demod\train_data',
                             mode_fun, f'data_{data_file_cnt}.csv')

    # 在Python中，使用numpy的genfromtxt或loadtxt函数来读取数据
    # 假设数据文件是CSV格式，如果不是，需要根据实际格式调整读取函数
    df = pd.read_csv(file_name, header=None, )

    signal_I_real = np.array(df.iloc[:, 0], dtype=np.float32)
    signal_Q_imag = np.array(df.iloc[:, 1], dtype=np.float32)

    # 在Python中，使用numpy的complex函数来创建复数数组
    signal_IQ = signal_I_real + 1j * signal_Q_imag

    # 参数读取
    Fs = 20e6  # 采样率20MHz
    N = len(signal_IQ)  # IQ数据长度
    t = (np.arange(N) / Fs)  # 时间序列
    f = np.arange(-Fs/2, Fs/2, Fs/N) + (Fs / N)    # 频率序列

    code_num = np.sum(~np.isnan( np.array(df.iloc[:, 2]) ))
    code = np.array(df.iloc[:code_num, 2], np.int32)

    code_width = float(df.iloc[0, 4] / 1e6)  # 码元宽度
    smp_per_code = int(round(code_width * Fs))
    sample_num = smp_per_code * code_num
    signal = np.zeros(sample_num, dtype=np.complex64)
    signal[:N] = signal_IQ

    code_r = np.tile(code.reshape((code.shape[0], 1)), (1, smp_per_code))
    code_t = code_r.reshape(1, sample_num)

    if N < sample_num:
        print('IQ data len is short than code_num * smp_per_code (sample_num).')
        t = np.arange(sample_num) / Fs  # 时间序列
        f = np.fft.fftfreq(sample_num, 1/Fs)  # 频率序列


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
    # 输出结果
    print(f'信号的频率范围为：{f_low:.2f} Hz - {f_high:.2f} Hz')
    print(f'信号的带宽为：{f_high - f_low:.2f} Hz')
    # 可视化
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(freq, cumulative_power)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Cumulative Power')
    plt.title('Cumulative Power Spectrum')
    plt.grid(True)

    plt.axhline(y=drop / 100, color='r', linestyle='--', label=f'{drop}% Power')
    plt.axhline(y=1 - (drop / 100), color='r', linestyle='--', label=f'{100-drop}% Power')
    plt.scatter([f_low, f_high], [drop/100, (100-drop)/100], color='r')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.grid(True)
    plt.plot(freq, np.abs(spectrum))

    plt.show()

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
    F_NCO = freq_mid / 1e6 - 0.04
    noc = np.exp(-1j * 2 * np.pi * (F_NCO * 1e6) * t)
    snf = signal * noc
    # sf_c = scipy.signal.lfilter(hc, [1], snf)
    # sf_rc = scipy.signal.lfilter(hrc, [1], snf)
    # sf_g = scipy.signal.lfilter(hg, [1], snf)
    # sf_lpf = scipy.signal.lfilter(hlpf, [1], snf)
    # signal_analy = np.column_stack((snf, sf_c, sf_rc, sf_g, sf_lpf))
    signal_analy = np.column_stack((snf, ))
    signal_analy_cnt = 1
    # signal_names = ['snf', 'sf_c', 'sf_rc', 'sf_g', 'sf_lpf']
    signal_names = ['snf', ]
    # 时域绘图
    plt.figure()
    for i in range(signal_analy_cnt):
        plt.subplot(signal_analy_cnt, 1, i + 1)
        plt.plot(t * 1e6, np.real(signal_analy[:, i]), 'r', t * 1e6, np.imag(signal_analy[:, i]), 'b')
        plt.xlabel('Time (us)')
        plt.ylabel('Amplitude')
        plt.title(f'Time Domain Signal: {signal_names[i]}')
        plt.grid(True)
    plt.show()
    # 相位绘图
    plt.figure()
    for i in range(signal_analy_cnt):
        plt.subplot(signal_analy_cnt, 1, i + 1)
        angle_t = np.angle(signal_analy[:, i])
        angle_t = np.where(angle_t < 0, angle_t + 2 * np.pi, angle_t)
        plt.plot(t * 1e6, angle_t, 'r')  # 真实相位
        plt.axvline(x=t[0:smp_per_code:sample_num] * 1e6, color='k', linestyle='--')  # 相位划分线
        plt.plot(t * 1e6, code_t.reshape(-1) * np.pi / 2 + np.pi / 4, 'k')  # 码元序列
        plt.xlabel('Time (us)')
        plt.ylabel('Amplitude')
        plt.title(f'Time Domain Signal: {signal_names[i]}')
        plt.grid(True)
    plt.show()
    # 频域绘图
    f = np.fft.fftfreq(len(signal_analy), d=1/Fs)
    plt.figure()
    # plt.plot(f / 1e6, 20 * np.log10(np.fft.fftshift(np.abs(np.fft.fft(signal_analy[:, 0])))))  # 转为MHz显示
    plt.plot(f / 1e6, 20 * np.log10(np.abs(np.fft.fft(signal_analy[:, 0]))))  # 转为MHz显示
    plt.axvline(x=-Fpass / 1e6, color='r', linewidth=2)
    plt.axvline(x=Fpass / 1e6, color='r', linewidth=2)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Magnitude')
    plt.title('Frequency Spectrum: NO Filter')
    plt.grid(True)
    plt.show()

    # 眼图
    for i in range(signal_analy_cnt):
        real_part = np.real(signal_analy[:, i])
        imag_part = np.imag(signal_analy[:, i])

        # Eye diagram
        # 这块的reshape为啥和matlab不一样？
        # eye_diagram_real = real_part.reshape(smp_per_code*2, -1)
        # eye_diagram_imag = imag_part.reshape(smp_per_code*2, -1)
        eye_diagram_real = real_part.reshape(-1, smp_per_code*2).T
        eye_diagram_imag = imag_part.reshape(-1, smp_per_code*2).T

        plt.figure()
        plt.subplot(signal_analy_cnt, 2, 2*i+1)   # real eye diagram
        plt.plot(eye_diagram_real, color='y')
        plt.gca().set_facecolor('k')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.title(f'Eye Diagram (Real): {signal_names[i]}')
        plt.grid(True)

        plt.subplot(signal_analy_cnt, 2, 2*i+2) # imag eye diagram
        plt.plot(eye_diagram_imag, color='y')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.title(f'Eye Diagram (Imag): {signal_names[i]}')
        plt.grid(True)
        plt.gca().set_facecolor('k')

        plt.show()

    # 星座图
    for i in range(signal_analy_cnt):
        plt.figure()
        for n in range(smp_per_code):  # Constellation diagram with different phases
            pha = n
            real_part = np.real(signal_analy[:, i]).T
            imag_part = np.imag(signal_analy[:, i]).T

            # Constellation diagram
            # 为什么这里的reshape和matlab中输出的维度一样，但值不一样？
            # scatter_diagram_real = real_part.reshape(smp_per_code, -1)
            # scatter_diagram_imag = imag_part.reshape(smp_per_code, -1)
            scatter_diagram_real = real_part.reshape(-1, smp_per_code).T
            scatter_diagram_imag = imag_part.reshape(-1, smp_per_code).T

            # plt.subplot(int(np.ceil(signal_analy_cnt/2)), 2, i+1) # Constellation diagram
            # plt.subplot(smp_per_code, signal_analy_cnt, (n*signal_analy_cnt + i)+1)  # Constellation diagram
            plt.subplot(math.ceil(math.sqrt(smp_per_code)),
                        round(math.sqrt(smp_per_code)),
                        n+1)  # Constellation diagram
            plt.plot(scatter_diagram_real[pha, :], scatter_diagram_imag[pha, :], 'y.')
            plt.xlabel('In-phase')
            plt.ylabel('Quadrature')
            plt.title(f'Constellation Diagram: {signal_names[i]}')
            plt.grid(True)
            plt.axis('equal')
        plt.show()

    print()

