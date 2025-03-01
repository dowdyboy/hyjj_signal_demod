import torch
import torch.nn as nn
import numpy as np
import os
import shutil
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from dowdyboy_lib.rand import set_seed
import dowdyboy_lib.log as logger
from dowdyboy_lib.model_util import save_checkpoint_unique

from hyjj.v7.dataset import HyjjDataset, HyjjDemodDataset
from hyjj.v7.model import UNET_1D


parser = argparse.ArgumentParser(description='train demod')
## model
parser.add_argument('--model-type', type=str, nargs='+', required=True, help='model type')
## data
parser.add_argument('--num-workers', type=int, default=4, help='num workers')
parser.add_argument('--data-dir', type=str, required=True, help='train data dir')
parser.add_argument('--data-len', type=int, default=2000, help='data len')
parser.add_argument('--data-pos', action='store_true', default=False, help='data pos')
parser.add_argument('--train-ratio', type=float, default=0.99, help='train ratio')
## optimizer
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--use-scheduler', action='store_true', default=False, help='use scheduler')
## train
parser.add_argument('--epoch-count', type=int, default=100, help='epoch count')
parser.add_argument('--val-per-epoch', type=int, default=1, help='val interval')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--device', type=str, default='cuda', help='device')
parser.add_argument('--seed', type=int, default=114514, help='random seed')
parser.add_argument('--output-dir', type=str, default='./output_v7_demod', help='out dir')
args = parser.parse_args()


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)


def build_data():
    data_list = []
    for model_name in args.model_type:
        train_base_ds = HyjjDataset(
            root_dir=args.data_dir,
            train_ratio=args.train_ratio,
            is_train=True,
            rand_seed=args.seed,
            label_filter=[model_name]
        )
        train_ds = HyjjDemodDataset(train_base_ds, args.data_len, args.data_pos)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, )

        val_base_ds = HyjjDataset(
            root_dir=args.data_dir,
            train_ratio=args.train_ratio,
            is_train=False,
            rand_seed=args.seed,
            label_filter=[model_name]
        )
        val_ds = HyjjDemodDataset(val_base_ds, args.data_len, args.data_pos)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False, )
        data_list.append(
            [train_dl, val_dl, train_ds, val_ds]
        )
    return data_list


def build_model():
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
    model_list = []
    for model_name in args.model_type:
        model_list.append(
            UNET_1D(3 if args.data_pos else 2,
                    num_class_map[model_name], 64, 7, 4)
        )
    return model_list


def build_optimizer(model_list):
    optimizer_list = []
    scheduler_list = []
    for model in model_list:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_list.append(
            optimizer
        )
        if args.use_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
            scheduler_list.append(
                scheduler
            )
        else:
            scheduler = None
    return optimizer_list, scheduler_list


def build_loss_func(model_list):
    loss_func_list = []
    for _ in model_list:
        loss_func_list.append(
            nn.CrossEntropyLoss()
        )
    return loss_func_list


def evaluate_model(model_demod, data_loader, criterion_demod, device):
    model_demod.eval()
    total_loss_demod = 0.
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels_demod in data_loader:
            inputs, labels_demod = to_device((inputs, labels_demod), device)

            outputs_demod = model_demod(inputs)
            loss_demod = criterion_demod(outputs_demod, labels_demod)
            total_loss_demod += loss_demod.item()

            predicted = torch.argmax(outputs_demod, dim=1)
            correct += (predicted == labels_demod).sum().item()
            total += torch.flatten(labels_demod).size(0)

            # all_labels.extend(labels_demod.cpu().numpy())
            # all_predictions.extend(predicted.cpu().numpy())

    avg_loss_demod = total_loss_demod / len(data_loader)
    accuracy = 100 * correct / total

    return avg_loss_demod, accuracy, all_labels, all_predictions


def main():
    set_seed(args.seed)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    logger.logging_conf(os.path.join(args.output_dir, 'runtime.log'))
    logger.log(args)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoint')

    data_list = build_data()
    for train_dl, val_dl, train_ds, val_ds in data_list:
        logger.log(f'TRAIN DATA COUNT: {len(train_ds)}, VAL DATA COUNT: {len(val_ds)}')
    model_list = build_model()

    optimizer_list, scheduler_list = build_optimizer(model_list)
    for i in range(len(optimizer_list)):
        if len(scheduler_list) > 0:
            logger.log(f'OPTIMIZER AMR: {optimizer_list[i]}, SCHEDULER AMR: {scheduler_list[i]}')
        else:
            logger.log(f'OPTIMIZER AMR: {optimizer_list[i]}')

    loss_func_list = build_loss_func(model_list)
    for loss_func in loss_func_list:
        logger.log(f'LOSS FUNC: {loss_func}')

    for model_idx, model_name in enumerate(args.model_type):
        model = model_list[model_idx]
        model = model.to(args.device)
        logger.log(f'MODEL Name: {model_name} Train Start...')
        train_dl, val_dl, train_ds, val_ds = data_list[model_idx]
        loss_func = loss_func_list[model_idx]
        optimizer_demod = optimizer_list[model_idx]
        if args.use_scheduler:
            scheduler_demod = scheduler_list[model_idx]

        best_val_acc = 0.0

        for ep in range(1, args.epoch_count+1):
            model.train()
            total_loss_demod = 0.
            with tqdm(total=len(train_dl), desc=f"Epoch [{ep}/{args.epoch_count}]", unit="batch") as pbar:
                for i, data in enumerate(train_dl):
                    bat_x, bat_y = data
                    bat_x = bat_x.to(args.device)
                    bat_y = bat_y.to(args.device)

                    pred_y = model(bat_x)
                    loss_demod = loss_func(pred_y, bat_y)
                    optimizer_demod.zero_grad()
                    loss_demod.backward()
                    optimizer_demod.step()
                    total_loss_demod += loss_demod.item()

                    pbar.update(1)
                    pbar.set_postfix(loss_demod=loss_demod.item(), )

                    if (i+1) % (len(train_dl) // args.val_per_epoch) == 0:
                        avg_loss_demod, accuracy, _, _ = evaluate_model(model, val_dl, loss_func, args.device)
                        logger.log(f'Validation Loss Demod: {avg_loss_demod:.4f}, Validation Accuracy: {accuracy:.2f}%')
                        if args.use_scheduler:
                            scheduler_demod.step(avg_loss_demod)
                        if accuracy > best_val_acc:
                            best_val_acc = accuracy
                            save_checkpoint_unique(f'{ep}_{i}', checkpoint_dir, [model], [optimizer_demod],
                                                   f'best_demod_{model_name}')
                            logger.log(f'Best Acc {model_name} Demod Model saved')
                        model.train()

            logger.log(f'Train Epoch [{ep}/{args.epoch_count}], '
                       f'Loss Demod: {total_loss_demod / len(train_dl):.4f},'
                       f'LR Demod: {optimizer_demod.param_groups[0].get("lr")}')

    print()


if __name__ == '__main__':
    main()


