"""
这是一个提交示例Python脚本，供选手参考。
-测评环境为Python3.8
-测评环境中提供了基础的包和框架，具体版本请查看【https://github.com/Datacastle-Algorithm-Department/images/blob/main/doc/py38.md】
-如有测评环境未安装的包，请在requirements.txt里列明, 最好列明版本，例如：numpy==1.23.5
-如不需要安装任何包，请保持requirements.txt文件为空即可，但是提交时一定要有此文件
"""

# 导入必要模块
import os
import sys
import random

import numpy as np
import pandas as pd
import torch

from hyjj.v6.model import CNNFeatureExtractor, BiLSTMModel2AMR, BiLSTMModel2CW
from hyjj.v6.dataset import process_nco_signal_infer, gen_amr_data_aug
from hyjj.v7.model import UNET_1D
from hyjj.v7.dataset import gen_demod_data_aug_infer


code_type_count = {
    1: 2,
    2: 4,
    3: 8,
    4: 2,
    5: 8,
    6: 16,
    7: 32,
    8: 8,
    9: 16,
    10: 32
}

mod_num_name_map = {
    1: 'BPSK',
    2: 'QPSK',
    3: '8PSK',
    4: 'MSK',
    5: '8QAM',
    6: '16QAM',
    7: '32QAM',
    8: '8APSK',
    9: '16APSK',
    10: '32APSK'
}


def build_model():
    return BiLSTMModel2AMR(
        CNNFeatureExtractor(hidden_chanel=16, num_layer=3, drop_rate=0.1),
        128, 128, 1,
    ), BiLSTMModel2CW(
        CNNFeatureExtractor(hidden_chanel=16, num_layer=3, drop_rate=0.1),
        128, 128, 1,
    )


def build_model_demod(data_pos):
    num_class_map = {
        'BPSK': 2,
        'QPSK': 4,
        '8PSK': 8,
        'MSK': 2,
        '8QAM': 8,
        '16QAM': 16,
        '32QAM': 32,
        '8APSK': 8,
        '16APSK': 16,
        '32APSK': 32
    }
    model_map = dict()
    for model_name in num_class_map.keys():
        model_map[model_name] = UNET_1D(3 if data_pos else 2,
                                        num_class_map[model_name], 64, 7, 4)
    return model_map


def gen_code_series(bat_x, amr, cw):
    code_len = bat_x.shape[2]
    count_per_sym = int(cw * 20)
    code_count = code_len // count_per_sym
    res = []
    for _ in range(code_count):
        res.append(random.randint(0, code_type_count[amr] - 1))
    return ' '.join(str(x) for x in res)


def infer_code_series(model, i_data_list, q_data_list, pos_data_list, data_len, code_width, is_pos, device):
    Fs = 20e6  # 采样率20MHz
    code_width = float(code_width / 1e6)
    smp_per_code = int(round(code_width * Fs))
    sample_len = len(i_data_list[0])

    ret = []
    for i in range(len(i_data_list)):
        cur_i, cur_q, cur_pos = i_data_list[i], q_data_list[i], pos_data_list[i]
        if is_pos:
            bat_x = torch.from_numpy(np.array([cur_i, cur_q, cur_pos], dtype=np.float32))
        else:
            bat_x = torch.from_numpy(np.array([cur_i, cur_q], dtype=np.float32))
        bat_x = torch.unsqueeze(bat_x, dim=0)

        with torch.no_grad():
            bat_x = bat_x.to(device)
            pred_y = model(bat_x)
        pred_y = torch.argmax(pred_y, dim=1).cpu().numpy()

        if i == len(i_data_list) - 1:
            left_data_len = data_len - i*sample_len
            for k in range(0, left_data_len, smp_per_code):
                c = np.argmax(np.bincount(pred_y[0, k:k+smp_per_code]))
                ret.append(c)
        else:
            for k in range(0, sample_len, smp_per_code):
                c = np.argmax(np.bincount(pred_y[0, k:k+smp_per_code]))
                ret.append(c)
    return ret


def main(to_pred_dir, result_save_path):
    """
    主函数，用于执行脚本的主要逻辑。

    参数：
        to_pred_dir: 测试集文件夹上层目录路径，不可更改!
        result_save_path: 预测结果文件保存路径，官方已指定为csv格式，不可更改!
    """
    # 获取测试集文件夹路径，不可更改！
    testpath = os.path.join(os.path.abspath(to_pred_dir), 'test')

    # 获取测试集文件列表
    test_file_lst = [name for name in os.listdir(testpath) if name.endswith('.csv')]

    # 初始化结果文件，定义表头，注意逗号之间无空格
    result = ['file_name,modulation_type,symbol_width,code_sequence']

    # 初始化模型
    device = 'cuda'
    amr_model_path = 'output_v7/checkpoint/chk_best_amr_step_91_2809/model_0.pt'
    cw_model_path = 'output_v7/checkpoint/chk_best_cw_step_34_2809/model_0.pt'
    use_sample_len = True
    sample_len = 2000
    demod_model_path = 'output_v7_demod/checkpoint'
    demod_data_pos = True
    demod_data_len = 2000

    model_amr, model_cw = build_model()
    model_amr.load_state_dict(torch.load(amr_model_path))
    model_cw.load_state_dict(torch.load(cw_model_path))
    model_amr = model_amr.to(device)
    model_cw = model_cw.to(device)
    model_amr.eval()
    model_cw.eval()

    demod_model_map = build_model_demod(demod_data_pos)
    demod_model_path_map = dict()
    for model_name_dir in os.listdir(demod_model_path):
        for model_name in demod_model_map.keys():
            if model_name in model_name_dir:
                demod_model_path_map[model_name] = os.path.join(demod_model_path, model_name_dir, 'model_0.pt')
                break
    for model_name in demod_model_map.keys():
        demod_model_map[model_name].load_state_dict(torch.load(demod_model_path_map[model_name]))
        demod_model_map[model_name] = demod_model_map[model_name].to(device)
        demod_model_map[model_name].eval()

    """
    读入测试集文件，调用模型进行预测。
    以下预测方式仅为示例参考，实现方式是用循环读入每个文件，依次进行预测，选手可以根据自己模型的情况自行修改
    """
    # 循环测试集文件列表对每个文件进行预测
    for filename in test_file_lst:
        # 待预测文件路径
        filepath = os.path.join(testpath, filename)

        # 读入测试文件，这里的测试文件为无表头的两列信号值
        df = pd.read_csv(filepath, header=None)
        i_signal, q_signal, spectrum = process_nco_signal_infer(df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy())
        i_signal_demod, q_signal_demod = np.copy(i_signal), np.copy(q_signal)
        ori_signal_len = len(i_signal_demod)
        if use_sample_len:
            i_signal, q_signal = gen_amr_data_aug(i_signal, q_signal, sample_len, )
        bat_x = torch.from_numpy(np.array([i_signal, q_signal], dtype=np.float32))
        bat_x = torch.unsqueeze(bat_x, dim=0)
        bat_x = bat_x.to(device)
        with torch.no_grad():
            pred_y_amr, pred_y_cw = model_amr(bat_x), model_cw(bat_x)
            pred_y_amr = int(torch.argmax(pred_y_amr, dim=1).cpu().numpy()[0]) + 1
            pred_y_cw = float(pred_y_cw[0].cpu().numpy())

        cur_model_name = mod_num_name_map[pred_y_amr]
        i_list, q_list, pos_list = gen_demod_data_aug_infer(i_signal_demod, q_signal_demod, demod_data_len, pred_y_cw)
        pred_cw_list = infer_code_series(demod_model_map[cur_model_name], i_list, q_list, pos_list,
                          ori_signal_len, pred_y_cw, demod_data_pos, device)


        """
        ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
        │加载模型，选手根据自己模型情况进行加载并进行多个任务的预测，在此不做展示│
        └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
        """

        # 以下三个需要预测指标分别指定了一个值代替预测值，选手需要根据自己模型进行实际的预测
        # 预测调制类型，数据类型为整型
        modulation_type = pred_y_amr

        # 预测码元宽度，数据类型为浮点型
        symbol_width = pred_y_cw

        # 预测码序列，数据类型为字符串，且序列内容为若干整数，数字之间用空格分隔
        # code_sequence = gen_code_series(bat_x.cpu().numpy(), modulation_type, symbol_width)
        code_sequence = ' '.join(str(x) for x in pred_cw_list)

        result.append("%s,%d,%f,%s" % (filename, modulation_type, symbol_width, code_sequence))

    # 将预测结果保存到result_save_path,保存方式可修改，但是注意保存路径不可更改！！！
    with open(result_save_path, 'w') as f:
        f.write('\n'.join(result))

    # 如果result为已经预测好的DataFrame数据，则可以直接使用pd.to_csv()的方式进行保存
    # result.to_csv(result_save_path, index=None)


if __name__ == "__main__":
    # ！！！以下内容不允许修改，修改会导致评分出错
    to_pred_dir = sys.argv[1]  # 官方给出的测试文件夹上层的路径，不可更改！
    result_save_path = sys.argv[2]  # 官方给出的预测结果保存文件路径，已指定格式为csv，不可更改！
    main(to_pred_dir, result_save_path)  # 运行main脚本，入参只有to_pred_dir, result_save_path，不可更改！

