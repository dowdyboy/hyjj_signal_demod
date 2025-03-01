import math
import pandas as pd
from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas
import random
import shutil

from hyjj.v6.dataset import HyjjDataset, process_nco_signal


def gen_dataset():
    mod_name_list = [
        'BPSK', 'QPSK', '8PSK', 'MSK', '8QAM',
        '16QAM', '32QAM', '8APSK', '16APSK', '32APSK'
    ]
    # mod_name_list = [
    #     'BPSK',
    # ]
    src_dir = r'E:\BaiduNetdiskDownload\hyjj_signal_demod\train_data'
    dst_dir = r'E:\BaiduNetdiskDownload\hyjj_signal_demod\demod_data'
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)

    for mod_name in mod_name_list:
        mod_dir = os.path.join(dst_dir, mod_name)
        if not os.path.isdir(mod_dir):
            os.makedirs(mod_dir)
        filename_prefix = 0

        base_ds = HyjjDataset(
            root_dir=src_dir,
            train_ratio=0.1,
            is_train=True,
            label_filter=[mod_name],
        )
        Fs = 20e6
        for i in range(len(base_ds)):
            i_data, q_data, code_series, amr_label, code_width = base_ds[i]
            i_data, q_data, spectrum = process_nco_signal(i_data, q_data, code_series.shape[0], code_series, code_width)
            smp_per_code = round(code_width / 1e6 * Fs)

            mod_length_dir = os.path.join(mod_dir, str(smp_per_code))
            if not os.path.isdir(mod_length_dir):
                os.makedirs(mod_length_dir)

            for code_i, sub_code in enumerate(list(code_series)):
                if code_i*smp_per_code >= len(i_data):
                    continue
                sub_i_data = i_data[code_i*smp_per_code:code_i*smp_per_code+smp_per_code]
                sub_q_data = q_data[code_i*smp_per_code:code_i*smp_per_code+smp_per_code]
                if len(sub_i_data) == smp_per_code:
                    save_filename = f'{filename_prefix}_{sub_code}.csv'
                    filename_prefix += 1
                    sub_iq = np.concatenate([sub_i_data.reshape(-1, 1), sub_q_data.reshape(-1, 1)], axis=1)
                    sub_df = pd.DataFrame(sub_iq)
                    sub_df.to_csv(
                        os.path.join(mod_length_dir, save_filename), index=False, header=False,
                    )


def split_val_dataset():
    src_dir = r'E:\BaiduNetdiskDownload\hyjj_signal_demod\demod_data'
    dst_dir = r'E:\BaiduNetdiskDownload\hyjj_signal_demod\demod_data_val'
    test_ratio = 0.1

    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)

    for type_name in os.listdir(src_dir):
        type_sub_dir = os.path.join(src_dir, type_name)
        for s_len in os.listdir(type_sub_dir):
            s_len_sub_dir = os.path.join(type_sub_dir, s_len)
            filename_list = os.listdir(s_len_sub_dir)
            filename_list = filename_list[:int(len(filename_list)*test_ratio)]
            for filename in filename_list:
                filepath = os.path.join(s_len_sub_dir, filename)
                dst_sub_dir = os.path.join(dst_dir, type_name, s_len)
                if not os.path.isdir(dst_sub_dir):
                    os.makedirs(dst_sub_dir)
                shutil.move(filepath, os.path.join(dst_sub_dir, filename))


if __name__ == '__main__':

    print()

