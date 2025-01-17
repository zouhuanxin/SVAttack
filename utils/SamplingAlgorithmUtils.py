import random
import sys

import numpy as np

import torch
import torch.nn.functional as F
import copy
from .ViewUtils import *


def crop_subsequence(input_data, num_of_frames, l_ratio=[0.95], output_size=64):
    C, T, V, M = input_data.shape

    if l_ratio[0] == 0.5:
        # if training , sample a random crop

        min_crop_length = 64
        scale = np.random.rand(1) * (l_ratio[1] - l_ratio[0]) + l_ratio[0]
        temporal_crop_length = np.minimum(np.maximum(int(np.floor(num_of_frames * scale)), min_crop_length),
                                          num_of_frames)

        start = np.random.randint(0, num_of_frames - temporal_crop_length + 1)
        temporal_crop = input_data[:, start:start + temporal_crop_length, :, :]

        temporal_crop = torch.tensor(temporal_crop, dtype=torch.float)
        temporal_crop = temporal_crop.permute(0, 2, 3, 1).contiguous().view(C * V * M, temporal_crop_length)
        temporal_crop = temporal_crop[None, :, :, None]
        temporal_crop = F.interpolate(temporal_crop, size=(output_size, 1), mode='bilinear', align_corners=False)
        temporal_crop = temporal_crop.squeeze(dim=3).squeeze(dim=0)
        temporal_crop = temporal_crop.contiguous().view(C, V, M, output_size).permute(0, 3, 1,
                                                                                      2).contiguous().numpy()

        return temporal_crop

    else:
        # if testing , sample a center crop

        start = int((1 - l_ratio[0]) * num_of_frames / 2)
        data = input_data[:, start:num_of_frames - start, :, :]
        temporal_crop_length = data.shape[1]

        temporal_crop = torch.tensor(data, dtype=torch.float)
        temporal_crop = temporal_crop.permute(0, 2, 3, 1).contiguous().view(C * V * M, temporal_crop_length)
        temporal_crop = temporal_crop[None, :, :, None]
        temporal_crop = F.interpolate(temporal_crop, size=(output_size, 1), mode='bilinear', align_corners=False)
        temporal_crop = temporal_crop.squeeze(dim=3).squeeze(dim=0)
        temporal_crop = temporal_crop.contiguous().view(C, V, M, output_size).permute(0, 3, 1,
                                                                                      2).contiguous().numpy()

        return temporal_crop


def Sampling(data_numpy):
    p_interval = [0.95]
    window_size = 64
    data_numpy = data_numpy.reshape(data_numpy.shape[1], data_numpy.shape[2], data_numpy.shape[3], data_numpy.shape[4])
    # data_numpy = data_numpy.cpu().numpy()
    valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
    if valid_frame_num == 0:  # 如果data_numpy为0说明这条数据异常，交给攻击算法去过滤掉这条数据
        return data_numpy
    data_numpy = valid_crop_resize(data_numpy, 64, p_interval, window_size)
    return data_numpy


def valid_crop_resize(data_numpy, valid_frame_num, p_interval, window):
    if window == -1:
        return data_numpy
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    # crop
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1 - p) * valid_size / 2)
        data = data_numpy[:, begin + bias:end - bias, :, :]  # center_crop
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size * p)), 64),
                                    valid_size)  # constraint cropped_length lower bound as 64
        bias = np.random.randint(0, valid_size - cropped_length + 1)
        data = data_numpy[:, begin + bias:begin + bias + cropped_length, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)
    # resize
    data = torch.tensor(data, dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, None, :, :]
    data = F.interpolate(data, size=(C * V * M, window), mode='bilinear', align_corners=False).squeeze()
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()
    data = data.reshape(1, data.shape[0], data.shape[1], data.shape[2], data.shape[3])
    # data = torch.tensor(data)
    return data


def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M 随机选择其中一段，不是很合理。因为有0
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def _rot(rot):
    """
    rot: T,3
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,3
    zeros = torch.zeros(rot.shape[0], 1)  # T,1
    ones = torch.ones(rot.shape[0], 1)  # T,1

    r1 = torch.stack((ones, zeros, zeros), dim=-1)  # T,1,3
    rx2 = torch.stack((zeros, cos_r[:, 0:1], sin_r[:, 0:1]), dim=-1)  # T,1,3
    rx3 = torch.stack((zeros, -sin_r[:, 0:1], cos_r[:, 0:1]), dim=-1)  # T,1,3
    rx = torch.cat((r1, rx2, rx3), dim=1)  # T,3,3

    ry1 = torch.stack((cos_r[:, 1:2], zeros, -sin_r[:, 1:2]), dim=-1)
    r2 = torch.stack((zeros, ones, zeros), dim=-1)
    ry3 = torch.stack((sin_r[:, 1:2], zeros, cos_r[:, 1:2]), dim=-1)
    ry = torch.cat((ry1, r2, ry3), dim=1)

    rz1 = torch.stack((cos_r[:, 2:3], sin_r[:, 2:3], zeros), dim=-1)
    r3 = torch.stack((zeros, zeros, ones), dim=-1)
    rz2 = torch.stack((-sin_r[:, 2:3], cos_r[:, 2:3], zeros), dim=-1)
    rz = torch.cat((rz1, rz2, r3), dim=1)

    rot = rz.matmul(ry).matmul(rx)
    return rot


def random_rot(data_numpy, theta=0.3):
    """
    data_numpy: C,T,V,M
    """
    data_torch = torch.from_numpy(data_numpy)
    C, T, V, M = data_torch.shape
    data_torch = data_torch.permute(1, 0, 2, 3).contiguous().view(T, C, V * M)  # T,3,V*M
    rot = torch.zeros(3).uniform_(-theta, theta)
    rot = torch.stack([rot, ] * T, dim=0)
    rot = _rot(rot)  # T,3,3
    data_torch = torch.matmul(rot, data_torch)
    data_torch = data_torch.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()

    return data_torch


def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
                                                             t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy


def Deal_ASGCN_data(tx):
    data_last = copy.copy(tx[:, -11:-10, :, :])
    target_data = copy.copy(tx[:, -10:, :, :])
    input_data = copy.copy(tx[:, :-10, :, :])
    valid_frame = (tx != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    length = end - begin
    if length <= 60:
        input_data_dnsp = input_data[:, :50, :, :]
    else:
        rs = int(np.random.uniform(low=0, high=np.ceil((length - 10) / 50)))
        input_data_dnsp = [input_data[:, int(i) + rs, :, :] for i in
                           [np.floor(j * ((length - 10) / 50)) for j in range(50)]]
        input_data_dnsp = np.array(input_data_dnsp).astype(np.float32)
        input_data_dnsp = np.transpose(input_data_dnsp, axes=(1, 0, 2, 3))
    input_data_dnsp = input_data_dnsp.reshape(1, input_data_dnsp.shape[0], input_data_dnsp.shape[1],
                                              input_data_dnsp.shape[2], input_data_dnsp.shape[3])
    input_data = input_data.reshape(1, input_data.shape[0], input_data.shape[1], input_data.shape[2],
                                    input_data.shape[3])
    target_data = target_data.reshape(1, target_data.shape[0], target_data.shape[1], target_data.shape[2],
                                      target_data.shape[3])
    data_last = data_last.reshape(1, data_last.shape[0], data_last.shape[1], data_last.shape[2],
                                  data_last.shape[3])
    input_data = torch.tensor(input_data).cuda()
    input_data_dnsp = torch.tensor(input_data_dnsp).cuda()
    target_data = torch.tensor(target_data).cuda()
    data_last = torch.tensor(data_last).cuda()

    return input_data, input_data_dnsp, target_data, data_last


def progress_bar(finish_tasks_number, tasks_number):
    percentage = round(finish_tasks_number / tasks_number * 100)
    print(
        "\r{}/{} 进度:{}%:".format(finish_tasks_number, tasks_number, percentage),
        "" * (percentage // 2), end="")
    sys.stdout.flush()


# 裁剪扰动数组
def ClipPertubData(tx, x):
    tx = torch.tensor(tx)
    x = torch.tensor(x)
    tx_2d = ThreeDimensionsToTwoDimensions(tx)
    x_2d = ThreeDimensionsToTwoDimensions(x)
    N, C, T, V, M = tx.shape

    for n in range(N):
        for t in range(T):
            for v in range(V):
                for m in range(M):
                    if torch.square(torch.sqrt(tx_2d[n, 0, t, v, m] - x_2d[n, 0, t, v, m]) + torch.sqrt(
                            tx_2d[n, 1, t, v, m] - x_2d[n, 1, t, v, m])) > 0.01:
                        x[n, 0, t, v, m] /= 2
                        x[n, 1, t, v, m] /= 2

    return x.numpy()
