import torch
import numpy as np
from .ViewUtils import *
from .DrawTools import *

parents = np.array([1, 1, 21, 3, 21,
                    5, 6, 7, 21, 9,
                    10, 11, 1, 13, 14,
                    15, 1, 17, 18, 19,
                    2, 8, 8, 12, 12]) - 1


def computer_pertual(save_path, device='cuda:0'):
    refData = np.load(f'{save_path}/samples_x_list.npy')
    adData = np.load(f'{save_path}/attck_samples_x_list.npy')

    refData = torch.tensor(refData).float().cuda(device)
    adData = torch.tensor(adData).float().cuda(device)
    M = len(adData)
    T = adData.shape[2]
    L = adData.shape[3]
    print(f'M={M} T={T} L={L}')

    ref_boneVecs = refData[:, :, :, :, :] - refData[:, :, :, parents, :] + 1e-8
    ad_boneVecs = adData[:, :, :, :, :] - adData[:, :, :, parents, :] + 1e-8

    deltaT = 1 / 30
    refAcc = (refData[:, :, 2:, :, 0] - 2 * refData[:, :, 1:-1, :, 0] + refData[:, :, :-2, :, 0]) / deltaT / deltaT
    adAcc = (adData[:, :, 2:, :, 0] - 2 * adData[:, :, 1:-1, :, 0] + adData[:, :, :-2, :, 0]) / deltaT / deltaT

    p = compute_delta_p(refData, adData, ref_boneVecs, ad_boneVecs, refAcc, adAcc, M, T, L)
    return p


def computer_pertual_2d(save_path, device='cuda:0'):
    refData = np.load(f'{save_path}/samples_x_list.npy')
    adData = np.load(f'{save_path}/attck_samples_x_list.npy', allow_pickle=False)
    refData = torch.tensor(refData).float().cuda(device)
    adData = torch.tensor(adData).float().cuda(device)

    M = len(adData)
    T = adData.shape[2]
    L = adData.shape[3]
    print(f'M={M} T={T} L={L}')

    refData = ThreeDimensionsToTwoDimensions(refData)
    adData = ThreeDimensionsToTwoDimensions(adData)

    ref_boneVecs = refData[:, :, :, :, :] - refData[:, :, :, parents, :] + 1e-8
    ad_boneVecs = adData[:, :, :, :, :] - adData[:, :, :, parents, :] + 1e-8

    deltaT = 1 / 30
    refAcc = (refData[:, 2:, :] - 2 * refData[:, 1:-1, :] + refData[:, :-2, :]) / deltaT / deltaT
    adAcc = (adData[:, 2:, :] - 2 * adData[:, 1:-1, :] + adData[:, :-2, :]) / deltaT / deltaT

    p = compute_delta_p(refData, adData, ref_boneVecs, ad_boneVecs, refAcc, adAcc, M, T, L)
    return p


def computer_pertual_2d_two(refData, adData, device='cuda:0'):
    refData = torch.tensor(refData).float().cuda(device)
    adData = torch.tensor(adData).float().cuda(device)

    M = len(adData)
    T = adData.shape[2]
    L = adData.shape[3]

    refData = ThreeDimensionsToTwoDimensions(refData)
    adData = ThreeDimensionsToTwoDimensions(adData)

    ref_boneVecs = refData[:, :, :, :, :] - refData[:, :, :, parents, :] + 1e-8
    ad_boneVecs = adData[:, :, :, :, :] - adData[:, :, :, parents, :] + 1e-8

    deltaT = 1 / 30
    refAcc = (refData[:, 2:, :] - 2 * refData[:, 1:-1, :] + refData[:, :-2, :]) / deltaT / deltaT
    adAcc = (adData[:, 2:, :] - 2 * adData[:, 1:-1, :] + adData[:, :-2, :]) / deltaT / deltaT

    p = compute_delta_p(refData, adData, ref_boneVecs, ad_boneVecs, refAcc, adAcc, M, T, L)
    return p


def computer_pertual_2d_for(save_path, start_angle=0, end_angle=360, interval_angle=10, device='cuda:0'):
    refData = np.load(f'{save_path}/samples_x_list.npy')
    adData = np.load(f'{save_path}/attck_samples_x_list.npy', allow_pickle=False)
    refData = torch.tensor(refData).float().cuda(device)
    adData = torch.tensor(adData).float().cuda(device)

    M = len(adData)
    T = adData.shape[2]
    L = adData.shape[3]
    print(f'M={M} T={T} L={L}')
    deltaT = 1 / 30
    elev = 0
    azim = 45
    p = []
    iter_num = int((end_angle - start_angle) / interval_angle)
    for i in range(iter_num + 1):
        refData_temp = ThreeDimensionsToTwoDimensions(refData, elev, azim)
        adData_temp = ThreeDimensionsToTwoDimensions(adData, elev, azim)

        ref_boneVecs_temp = refData_temp[:, :, :, :, :] - refData_temp[:, :, :, parents, :] + 1e-8
        ad_boneVecs_temp = adData_temp[:, :, :, :, :] - adData_temp[:, :, :, parents, :] + 1e-8

        refAcc_temp = (refData_temp[:, 2:, :] - 2 * refData_temp[:, 1:-1, :] + refData_temp[:, :-2,
                                                                               :]) / deltaT / deltaT
        adAcc_temp = (adData_temp[:, 2:, :] - 2 * adData_temp[:, 1:-1, :] + adData_temp[:, :-2, :]) / deltaT / deltaT

        p.append(
            compute_delta_p(refData_temp, adData_temp, ref_boneVecs_temp, ad_boneVecs_temp, refAcc_temp, adAcc_temp, M,
                            T, L))
        elev += interval_angle
    return p


def computer_variance(save_path, device='cuda:0'):
    refData = np.load(f'{save_path}/samples_x_list.npy')
    adData = np.load(f'{save_path}/attck_samples_x_list.npy', allow_pickle=False)
    refData = torch.tensor(refData).float().cuda(device)
    adData = torch.tensor(adData).float().cuda(device)

    diffData = adData - refData
    std = torch.std(diffData, dim=0)
    return torch.sum(std)


def compute_delta_p(S, S_hat, B, B_hat, S_dot, S_dot_hat, M, T, L):
    term1 = torch.norm(S - S_hat, p=2, dim=-1).sum() / (M * T)
    term2 = torch.norm(B - B_hat, p=2, dim=-1).sum() / (M * T)
    term3 = torch.norm(S_dot - S_dot_hat, p=2, dim=-1).sum() / (M * T * L)
    # term3 = S_dot_hat.sum() / (M * T * L)

    delta_p = term1 + term2 + term3
    return delta_p.item()

