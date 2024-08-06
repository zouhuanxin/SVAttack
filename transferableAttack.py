import numpy as np
import torch
import argparse
from omegaconf import OmegaConf
from utils.loadModel import *

def qyl(target_classifier, save_path, device='cuda:0'):
    source_x_list = np.load(f'{save_path}/source_x_list.npy')
    frames_list = np.load(f'{save_path}/frames_list.npy')
    attck_x_list = np.load(f'{save_path}/attck_x_list.npy', allow_pickle=False)
    source_y_list = np.load(f'{save_path}/source_y_list.npy')
    f_num = 0
    sample_num = len(attck_x_list)
    for i in range(len(attck_x_list)):
        tx = source_x_list[i]
        tx = tx.reshape(1, tx.shape[0], tx.shape[1], tx.shape[2], tx.shape[3])
        tx = torch.tensor(tx).float().cuda(device)

        x = attck_x_list[i]
        x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        x = torch.tensor(x).float().cuda(device)

        pred = target_classifier(x)
        predictedLabels = torch.argmax(pred, axis=1)
        if source_y_list[i] != predictedLabels and torch.all(tx == 0) == False:
            f_num += 1
        if torch.all(tx == 0):  # 过滤异常数据
            sample_num -= 1
    print(f'迁移率 {f_num / sample_num} 当前对抗样本数量 {sample_num}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/stgcn.yaml")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    qyl(getModel('agcn'), config.save_path)
