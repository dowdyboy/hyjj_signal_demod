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

from hyjj.v3.model import CNNFeatureExtractor, BiLSTMModel2
from hyjj.v3.dataset import process_nco_signal_infer


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


def build_model():
    return BiLSTMModel2(
        CNNFeatureExtractor(hidden_chanel=16, num_layer=3, drop_rate=0.1),
        128, 128, 1,
    )


def gen_code_series(bat_x, amr, cw):
    code_len = bat_x.shape[2]
    count_per_sym = int(cw * 20)
    code_count = code_len // count_per_sym
    res = []
    for _ in range(code_count):
        res.append(random.randint(0, code_type_count[amr] - 1))
    return ' '.join(str(x) for x in res)


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
    amr_model_path = 'output_v2/checkpoint/chk_best_amr_step_67_1265/model_0.pt'
    cw_model_path = 'output_v2/checkpoint/chk_best_cw_step_58_1265/model_0.pt'
    model_amr = build_model()
    model_cw = build_model()
    model_amr.load_state_dict(torch.load(amr_model_path))
    model_cw.load_state_dict(torch.load(cw_model_path))
    model_amr = model_amr.to(device)
    model_cw = model_cw.to(device)
    model_amr.eval()
    model_cw.eval()

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
        bat_x = torch.from_numpy(np.array([i_signal, q_signal], dtype=np.float32))
        bat_x = torch.unsqueeze(bat_x, dim=0)
        bat_x = bat_x.to(device)
        with torch.no_grad():
            pred_y_amr, pred_y_cw = model_amr(bat_x), model_cw(bat_x)
            pred_y_amr = pred_y_amr[0]
            pred_y_cw = pred_y_cw[1]
            pred_y_amr = int(torch.argmax(pred_y_amr, dim=1).cpu().numpy()[0]) + 1
            pred_y_cw = float(pred_y_cw[0].cpu().numpy())

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
        code_sequence = gen_code_series(bat_x.cpu().numpy(), modulation_type, symbol_width)

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

