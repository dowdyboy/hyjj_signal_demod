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

from hyjj.v5.dataset import HyjjDataset, HyjjAMRCWDataset
from hyjj.v5.model import BiLSTMModel, CNNFeatureExtractor, BiLSTMModel2

parser = argparse.ArgumentParser(description='train radar')
## model
## data
parser.add_argument('--num-workers', type=int, default=4, help='num workers')
parser.add_argument('--data-dir', type=str, required=True, help='data dir')
parser.add_argument('--train-ratio', type=float, default=0.9, help='train ratio')
parser.add_argument('--sample-len', type=int, default=1560, help='data sample len')
parser.add_argument('--flip-rate', type=float, default=0.5, help='data flip rate')
## optimizer
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--use-scheduler', action='store_true', default=False, help='use scheduler')
## train
parser.add_argument('--epoch-count', type=int, default=100, help='epoch count')
parser.add_argument('--val-per-epoch', type=int, default=1, help='val interval')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--device', type=str, default='cuda', help='device')
parser.add_argument('--seed', type=int, default=114514, help='random seed')
parser.add_argument('--output-dir', type=str, default='./output_v5', help='out dir')
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
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False, )
    return train_dl, val_dl, train_ds, val_ds


def build_model():
    # return BiLSTMModel(CNNFeatureExtractor())
    # return UNET_1D(2, 1, 64, 7, 3, args.sample_len)

    # return BiLSTMModel(
    #     CNNFeatureExtractor(hidden_chanel=16, num_layer=3, drop_rate=0.1),
    #     128, 128, 1,
    # )

    return BiLSTMModel2(
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
    # return nn.CrossEntropyLoss()
    return nn.CrossEntropyLoss(), nn.L1Loss()


# def evaluate_model(model, data_loader, criterion, device):
#     model.eval()
#     total_loss = 0.0
#     correct = 0
#     total = 0
#     all_labels = []
#     all_predictions = []
#
#     with torch.no_grad():
#         for inputs, labels in data_loader:
#             inputs, labels = to_device((inputs, labels), device)
#
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()
#
#             predicted = torch.argmax(outputs, dim=1)
#             correct += (predicted == labels).sum().item()
#             total += labels.size(0)
#
#             all_labels.extend(labels.cpu().numpy())
#             all_predictions.extend(predicted.cpu().numpy())
#
#     avg_loss = total_loss / len(data_loader)
#     accuracy = 100 * correct / total
#
#     return avg_loss, accuracy, all_labels, all_predictions


def evaluate_model(model, data_loader, criterion_amr, criterion_cw, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_cw_labels = []
    all_cw_predictions = []

    with torch.no_grad():
        for inputs, labels_amr, labels_cw in data_loader:
            inputs, labels_amr, labels_cw = to_device((inputs, labels_amr, labels_cw), device)

            outputs_amr, outputs_cw = model(inputs)
            loss = criterion_amr(outputs_amr, labels_amr) * 1.0 + criterion_cw(outputs_cw, labels_cw) * 1.0
            total_loss += loss.item()

            predicted = torch.argmax(outputs_amr, dim=1)
            correct += (predicted == labels_amr).sum().item()
            total += labels_amr.size(0)

            all_labels.extend(labels_amr.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            all_cw_labels.extend(labels_cw.cpu().numpy())
            all_cw_predictions.extend(outputs_cw.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total

    cw_mean = float(np.mean(np.abs(np.array(all_cw_predictions) - np.array(all_cw_labels))))

    return avg_loss, accuracy, cw_mean, all_labels, all_predictions


def main():
    set_seed(args.seed)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    logger.logging_conf(os.path.join(args.output_dir, 'runtime.log'))
    logger.log(args)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoint')

    train_dl, val_dl, train_ds, val_ds = build_data()
    logger.log(f'TRAIN DATA COUNT: {len(train_ds)}, VAL DATA COUNT: {len(val_ds)}')
    model = build_model()
    model = model.to(args.device)
    logger.log(f'MODEL: {model}')
    optimizer, scheduler = build_optimizer(model)
    logger.log(f'OPTIMIZER: {optimizer}, SCHEDULER: {scheduler}')
    # loss_func = build_loss_func()
    # logger.log(f'LOSS FUNC: {loss_func}')
    loss_func_amr, loss_func_cw = build_loss_func()
    logger.log(f'LOSS FUNC: {loss_func_amr}, {loss_func_cw}')

    best_val_acc = 0.0
    best_val_cw_mean = 10.0
    avg_val_loss = 0.0

    for ep in range(1, args.epoch_count+1):
        model.train()
        total_loss = 0
        with tqdm(total=len(train_dl), desc=f"Epoch [{ep}/{args.epoch_count}]", unit="batch") as pbar:
            for i, data in enumerate(train_dl):
                bat_x, bat_y_amr, bat_y_cw = data
                bat_x = bat_x.to(args.device)
                # bat_y = bat_y.to(args.device)
                bat_y_amr = bat_y_amr.to(args.device)
                bat_y_cw = bat_y_cw.to(args.device)
                pred_y_amr, pred_y_cw = model(bat_x)
                loss = loss_func_amr(pred_y_amr, bat_y_amr) * 1.0 + loss_func_cw(pred_y_cw, bat_y_cw) * 1.0
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

                if (i+1) % (len(train_dl) // args.val_per_epoch) == 0:
                    avg_loss, accuracy, cw_mean, _, _ = evaluate_model(model, val_dl, loss_func_amr, loss_func_cw, args.device)
                    avg_val_loss = avg_loss
                    logger.log(f'Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%, CW Mean: {cw_mean}')
                    if args.use_scheduler:
                        scheduler.step(avg_val_loss)
                    if accuracy > best_val_acc:
                        best_val_acc = accuracy
                        save_checkpoint_unique(f'{ep}_{i}', checkpoint_dir, [model], [optimizer], 'best_amr')
                        logger.log(f'Best Acc AMR Model saved')
                    if cw_mean < best_val_cw_mean:
                        best_val_cw_mean = cw_mean
                        save_checkpoint_unique(f'{ep}_{i}', checkpoint_dir, [model], [optimizer], 'best_cw')
                        logger.log(f'Best Acc CW Model saved')
                    model.train()

        logger.log(f'Train Epoch [{ep}/{args.epoch_count}], Loss: {total_loss / len(train_dl):.4f}, LR: {optimizer.param_groups[0].get("lr")}')

        # avg_loss, accuracy, _, _ = evaluate_model(model, val_dl, loss_func, args.device)
        # logger.log(f'Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')

    print()


if __name__ == '__main__':
    main()

