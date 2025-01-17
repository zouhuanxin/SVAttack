import torch
import torch as K
import numpy as np

parents = np.array([10, 0, 1, 2, 3,
                         10, 5, 6, 7, 8,
                         10, 10, 11, 12, 13,
                         13, 15, 16, 17, 18,
                         13, 20, 21, 22, 23])
jointWeights = torch.Tensor([[[0.02, 0.02, 0.02, 0.02, 0.02,
                                    0.02, 0.02, 0.02, 0.02, 0.02,
                                    0.04, 0.04, 0.04, 0.04, 0.04,
                                    0.02, 0.02, 0.02, 0.02, 0.02,
                                    0.02, 0.02, 0.02, 0.02, 0.02]]]).cuda()
deltaT = 1 / 30
reconWeight = 0.4
boneLenWeight = 0.7
perpLossType = 'l2_acc-bone'


def reshapeData(x, toNative=True):  # 调整数据格式
    if toNative:
        x = x.permute(0, 2, 3, 1, 4)
        x = x.reshape((x.shape[0], x.shape[1], -1, x.shape[4]))
    else:
        x = x.reshape((x.shape[0], x.shape[1], -1, 3, x.shape[4]))
        x = x.permute(0, 3, 1, 2, 4)
    return x


def boneLengths(data):

    jpositions = K.reshape(data, (data.shape[0], data.shape[1], -1, int(data.shape[2] / 25)))

    boneVecs = jpositions - jpositions[:, :, parents, :] + 1e-8

    boneLengths = torch.sqrt(torch.sum(torch.square(boneVecs), axis=-1))

    return boneLengths

def boneLengthLoss(parentIds, adData, refBoneLengths):

    jpositions = K.reshape(adData, (adData.shape[0], adData.shape[1], -1, int(adData.shape[2] / 25)))

    boneVecs = jpositions - jpositions[:, :, parents, :] + 1e-8

    boneLengths = torch.sqrt(torch.sum(torch.square(boneVecs), axis=-1))

    boneLengthsLoss = K.mean(
        K.sum(K.sum(K.square(boneLengths - refBoneLengths), axis=-1), axis=-1))
    return boneLengthsLoss

def accLoss(adData, refData, jointWeights=None):
    refAcc = (refData[:, 2:, :] - 2 * refData[:, 1:-1, :] + refData[:, :-2, :]) / deltaT / deltaT

    adAcc = (adData[:, 2:, :] - 2 * adData[:, 1:-1, :] + adData[:, :-2, :]) / deltaT / deltaT

    if jointWeights == None:
        return K.mean(K.sum(K.sum(K.square(adAcc - refAcc), axis=-1), axis=-1), axis=-1)
    else:
        return K.mean(K.sum(K.sum(K.square(adAcc - refAcc), axis=-1), axis=-1) * jointWeights, axis=-1)

def perceptualLoss(refData, adData, refBoneLengths):

    elements = perpLossType.split('_')

    if elements[0] == 'l2' or elements[0] == 'l2Clip':

        diffmx = K.square(refData - adData),
        squaredLoss = K.sum(K.reshape(K.square(refData - adData), (refData.shape[0], refData.shape[1], 25, -1)),
                            axis=-1)

        weightedSquaredLoss = squaredLoss * jointWeights

        squareCost = K.sum(K.sum(weightedSquaredLoss, axis=-1), axis=-1)

        oloss = K.mean(squareCost, axis=-1)



    elif elements[0] == 'lInf':
        squaredLoss = K.sum(K.reshape(K.square(refData - adData), (refData.shape[0], refData.shape[1], 25, -1)),
                            axis=-1)

        weightedSquaredLoss = squaredLoss * jointWeights

        squareCost = K.sum(weightedSquaredLoss, axis=-1)

        oloss = K.mean(K.norm(squareCost, ord=np.inf, axis=0))

    else:
        print('warning: no reconstruction loss')
        return

    if len(elements) == 1:
        return oloss

    elif elements[1] == 'acc-bone':

        jointAcc = accLoss(adData, refData)

        boneLengthsLoss = boneLengthLoss(parents, adData, refBoneLengths)

        return boneLengthsLoss * (1 - reconWeight) * boneLenWeight + jointAcc * (1 - reconWeight) * (
                1 - boneLenWeight) + oloss * reconWeight


def computer_perceptual_loss(tx, adData):
    if len(tx.shape) > 3:
        convertedData = reshapeData(tx)
        convertedAdData = reshapeData(adData)
        if len(convertedData.shape) > 3:
            percepLoss = 0
            for i in range(convertedData.shape[-1]):
                bLengths = boneLengths(convertedData[:, :, :, i])
                percepLoss += perceptualLoss(convertedData[:, :, :, i], convertedAdData[:, :, :, i], bLengths)
        else:
            bLengths = boneLengths(convertedData)
            percepLoss = perceptualLoss(convertedData, convertedAdData, bLengths)
    else:
        bLengths = boneLengths(tx)
        percepLoss = perceptualLoss(tx, adData, bLengths)

    return percepLoss
