import os
import numpy as np


def writer(samples_x_list, frames_list, attck_samples_x_list, attck_samples_y_list, save_path):
    np.save(f'{save_path}/samples_x_list.npy', np.stack(samples_x_list))
    np.save(f'{save_path}/frames_list.npy', np.stack(frames_list))
    np.save(f'{save_path}/attck_samples_x_list.npy', np.stack(attck_samples_x_list))
    np.save(f'{save_path}/attck_samples_y_list.npy', np.stack(attck_samples_y_list))
    print(f'当前样本数量={len(attck_samples_x_list)}')


def read(save_path):
    samples_x_list = np.load(f'{save_path}/samples_x_list.npy').tolist()
    frames_list = np.load(f'{save_path}/frames_list.npy').tolist()
    attck_samples_x_list = np.load(f'{save_path}/attck_samples_x_list.npy').tolist()
    attck_samples_y_list = np.load(f'{save_path}/attck_samples_y_list.npy').tolist()

    return samples_x_list, frames_list, attck_samples_x_list, attck_samples_y_list