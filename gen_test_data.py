import os
import pandas as pd

#
# \train_data目录下为BPSK、QPSK等子目录
#


if __name__ == '__main__':
    src_dir = r'E:\BaiduNetdiskDownload\hyjj_signal_demod\train_data'
    dst_dir = r'E:\BaiduNetdiskDownload\hyjj_signal_demod\test_data'
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
    for dirname in os.listdir(src_dir):
        file_list = os.listdir(os.path.join(src_dir, dirname))
        file_list = file_list[:1000]
        for filename in file_list:
            df = pd.read_csv(os.path.join(src_dir, dirname, filename), header=None, )
            df = df.iloc[:, 0:2]
            df.to_csv(os.path.join(dst_dir, f'{dirname}_{filename}'), header=None, index=None)

    print()
