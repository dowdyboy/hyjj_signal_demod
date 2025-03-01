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

from hyjj.v6.dataset import HyjjDataset, HyjjAMRCWDataset
from hyjj.v6.model import CNNFeatureExtractor, BiLSTMModel2AMR, BiLSTMModel2CW

parser = argparse.ArgumentParser(description='train radar')
## model
## data
parser.add_argument('--num-workers', type=int, default=4, help='num workers')
parser.add_argument('--data-dir', type=str, required=True, help='data dir')
parser.add_argument('--train-ratio', type=float, default=0.999, help='train ratio')
parser.add_argument('--sample-len', type=int, default=2000, help='data sample len')
parser.add_argument('--flip-rate', type=float, default=0.5, help='data flip rate')
## optimizer
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--use-scheduler', action='store_true', default=False, help='use scheduler')
## train
parser.add_argument('--epoch-count', type=int, default=100, help='epoch count')
parser.add_argument('--val-per-epoch', type=int, default=1, help='val interval')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--device', type=str, default='cuda', help='device')
parser.add_argument('--seed', type=int, default=114514, help='random seed')
parser.add_argument('--output-dir', type=str, default='./output_v7', help='out dir')
args = parser.parse_args()


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)


def build_data():
    train_base_ds = HyjjDataset(
        root_dir=args.data_dir,
        train_ratio=args.train_ratio,
        is_train=True,
        rand_seed=args.seed
    )
    val_base_ds = HyjjDataset(
        root_dir=args.data_dir,
        train_ratio=args.train_ratio,
        is_train=False,
        rand_seed=args.seed
    )
    train_ds = HyjjAMRCWDataset(train_base_ds, args.sample_len, sample_type='rand', flip_rate=args.flip_rate, )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, )
    val_ds = HyjjAMRCWDataset(val_base_ds, args.sample_len, )
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False, )
    return train_dl, val_dl, train_ds, val_ds


def build_model():
    # return BiLSTMModel(CNNFeatureExtractor())
    # return UNET_1D(2, 1, 64, 7, 3, args.sample_len)

    # return BiLSTMModel(
    #     CNNFeatureExtractor(hidden_chanel=16, num_layer=3, drop_rate=0.1),
    #     128, 128, 1,
    # )

    return BiLSTMModel2AMR(
        CNNFeatureExtractor(hidden_chanel=16, num_layer=3, drop_rate=0.1),
        128, 128, 1,
    ), BiLSTMModel2CW(
        CNNFeatureExtractor(hidden_chanel=16, num_layer=3, drop_rate=0.1),
        128, 128, 1,
    )


def build_optimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    else:
        scheduler = None
    return optimizer, scheduler


def build_loss_func():
    return nn.CrossEntropyLoss(), nn.L1Loss()


def evaluate_model(model_amr, model_cw, data_loader, criterion_amr, criterion_cw, device):
    model_amr.eval()
    model_cw.eval()
    total_loss_amr = 0.
    total_loss_cw = 0.
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_cw_labels = []
    all_cw_predictions = []

    with torch.no_grad():
        for inputs, labels_amr, labels_cw in data_loader:
            inputs, labels_amr, labels_cw = to_device((inputs, labels_amr, labels_cw), device)

            outputs_amr = model_amr(inputs)
            loss_amr = criterion_amr(outputs_amr, labels_amr)
            total_loss_amr += loss_amr.item()

            outputs_cw = model_cw(inputs)
            loss_cw = criterion_cw(outputs_cw, labels_cw)
            total_loss_cw += loss_cw.item()

            predicted = torch.argmax(outputs_amr, dim=1)
            correct += (predicted == labels_amr).sum().item()
            total += labels_amr.size(0)

            all_labels.extend(labels_amr.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            all_cw_labels.extend(labels_cw.cpu().numpy())
            all_cw_predictions.extend(outputs_cw.cpu().numpy())

    avg_loss_amr = total_loss_amr / len(data_loader)
    avg_loss_cw = total_loss_cw / len(data_loader)
    accuracy = 100 * correct / total

    cw_mean = float(np.mean(np.abs(np.array(all_cw_predictions) - np.array(all_cw_labels))))

    return avg_loss_amr, avg_loss_cw, accuracy, cw_mean, all_labels, all_predictions


def main():
    set_seed(args.seed)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    logger.logging_conf(os.path.join(args.output_dir, 'runtime.log'))
    logger.log(args)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoint')

    train_dl, val_dl, train_ds, val_ds = build_data()
    logger.log(f'TRAIN DATA COUNT: {len(train_ds)}, VAL DATA COUNT: {len(val_ds)}')
    model_amr, model_cw = build_model()
    model_amr = model_amr.to(args.device)
    model_cw = model_cw.to(args.device)
    logger.log(f'MODEL AMR: {model_amr}')
    logger.log(f'MODEL CW: {model_cw}')
    optimizer_amr, scheduler_amr = build_optimizer(model_amr)
    optimizer_cw, scheduler_cw = build_optimizer(model_cw)
    logger.log(f'OPTIMIZER AMR: {optimizer_amr}, SCHEDULER AMR: {scheduler_amr}')
    logger.log(f'OPTIMIZER CW: {optimizer_cw}, SCHEDULER CW: {scheduler_cw}')
    # loss_func = build_loss_func()
    # logger.log(f'LOSS FUNC: {loss_func}')
    loss_func_amr, loss_func_cw = build_loss_func()
    logger.log(f'LOSS FUNC: {loss_func_amr}, {loss_func_cw}')

    best_val_acc = 0.0
    best_val_cw_mean = 10.0

    for ep in range(1, args.epoch_count+1):
        model_amr.train()
        model_cw.train()
        total_loss_amr = 0.
        total_loss_cw = 0.
        with tqdm(total=len(train_dl), desc=f"Epoch [{ep}/{args.epoch_count}]", unit="batch") as pbar:
            for i, data in enumerate(train_dl):
                bat_x, bat_y_amr, bat_y_cw = data
                bat_x = bat_x.to(args.device)
                bat_y_amr = bat_y_amr.to(args.device)
                bat_y_cw = bat_y_cw.to(args.device)

                pred_y_amr = model_amr(bat_x)
                loss_amr = loss_func_amr(pred_y_amr, bat_y_amr)
                optimizer_amr.zero_grad()
                loss_amr.backward()
                optimizer_amr.step()
                total_loss_amr += loss_amr.item()

                pred_y_cw = model_cw(bat_x)
                loss_cw = loss_func_cw(pred_y_cw, bat_y_cw)
                optimizer_cw.zero_grad()
                loss_cw.backward()
                optimizer_cw.step()
                total_loss_cw += loss_cw.item()

                pbar.update(1)
                pbar.set_postfix(loss_amr=loss_amr.item(), loss_cw=loss_cw.item())

                if (i+1) % (len(train_dl) // args.val_per_epoch) == 0:
                    avg_loss_amr, avg_loss_cw, accuracy, cw_mean, _, _ = evaluate_model(model_amr, model_cw, val_dl, loss_func_amr, loss_func_cw, args.device)
                    logger.log(f'Validation Loss AMR: {avg_loss_amr:.4f}, Validation Accuracy: {accuracy:.2f}%, '
                               f'Validation Loss CW: {avg_loss_cw:.4f}, CW Mean: {cw_mean}')
                    if args.use_scheduler:
                        scheduler_amr.step(avg_loss_amr)
                        scheduler_cw.step(avg_loss_cw)
                    if accuracy > best_val_acc:
                        best_val_acc = accuracy
                        save_checkpoint_unique(f'{ep}_{i}', checkpoint_dir, [model_amr], [optimizer_amr], 'best_amr')
                        logger.log(f'Best Acc AMR Model saved')
                    if cw_mean < best_val_cw_mean:
                        best_val_cw_mean = cw_mean
                        save_checkpoint_unique(f'{ep}_{i}', checkpoint_dir, [model_cw], [optimizer_cw], 'best_cw')
                        logger.log(f'Best Acc CW Model saved')
                    model_amr.train()
                    model_cw.train()

        logger.log(f'Train Epoch [{ep}/{args.epoch_count}], '
                   f'Loss AMR: {total_loss_amr / len(train_dl):.4f}, Loss CW: {total_loss_cw / len(train_dl):.4f}, '
                   f'LR AMR: {optimizer_amr.param_groups[0].get("lr")}, LR CW: {optimizer_cw.param_groups[0].get("lr")}')

        # avg_loss, accuracy, _, _ = evaluate_model(model, val_dl, loss_func, args.device)
        # logger.log(f'Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')

    print()


if __name__ == '__main__':
    main()

