import os

import torch

from utils.loadModel import *
from utils.PerceptualUtils import *
from utils.MiddleUtils import *
from utils.SamplingAlgorithmUtils import *
import utils.FileUtils as FileUtil
from utils.PerceptualLoss import *
from torch.utils.data import DataLoader
from feeder.feeder import *
from feeder.ctrgcn.ctrgcn_feeder_ntu import Feeder as ctrgcn_feeder
import argparse
from omegaconf import OmegaConf


class Attacker():
    def __init__(self, args):
        super().__init__()
        self.name = 'SingleViewAttack'
        print(
            f'运行参数 {args}')
        self.updateClip = args.updateClip
        self.classWeight = args.classWeight
        self.epochs = args.epochs
        self.updateRule = args.updateRule
        self.model_name = args.model_name
        self.attackType = args.attackType
        self.save_path = args.save_path
        self.classifier = nn.DataParallel(getModel(args.model_name))
        self.end_sample_num = args.end_sample_num
        self.class_num = args.class_num
        self.batch_size = args.batch_size
        self.auto_learn_ratio = args.auto_learn_ratio
        self.perceptual_delta = args.perceptual_delta
        self.clamp_delta = args.clamp_delta
        self.perceptualType = args.perceptualType

        data_path = args.data_path
        if self.model_name == 'ctrgcn' or self.model_name == 'ctrgcn120':
            if self.auto_learn_ratio:
                self.resetAutoLearnRatio()
            else:
                if self.updateRule == 'gd':
                    print('gd学习率')
                    self.learningRate = 0.01
                else:
                    print('ifgsm学习率')
                    self.learningRate = 0.0001
            feeder = ctrgcn_feeder(data_path, p_interval=[0.95], window_size=64)
        else:
            if self.auto_learn_ratio:
                self.resetAutoLearnRatio()
            else:
                if self.updateRule == 'gd':
                    print('gd学习率')
                    self.learningRate = 0.01
                else:
                    print('ifgsm学习率')
                    self.learningRate = 0.0001
            label_path = args.label_path
            num_frame_path = args.num_frame_path
            feeder = Feeder(data_path, label_path, num_frame_path)
        self.trainloader = DataLoader(feeder,
                                      batch_size=args.batch_size,
                                      num_workers=4, pin_memory=True, drop_last=True, shuffle=False)
        if args.hookstatus:
            register_hooks(self.classifier, args.hookratio)

    def resetAutoLearnRatio(self):
        print('自动学习率')
        if self.model_name == 'ctrgcn' or self.model_name == 'ctrgcn120':
            self.learningRate = torch.ones([self.batch_size, 3, 64, 25, 2]).cuda() * 0.001
        else:
            self.learningRate = torch.ones([self.batch_size, 3, 300, 25, 2]).cuda() * 0.001

    def foolRateCal(self, rlabels, flabels):  # 计算欺骗率，即成功攻击的样本所占比例
        hitIndices = []
        for i in range(0, len(flabels)):
            if flabels[i] != rlabels[i]:
                hitIndices.append(i)
        return len(hitIndices) / len(flabels) * 100

    def getUpdate(self, grads, input):  # 获取更新后的对抗样本
        if self.updateRule == 'gd':

            return input - grads * self.learningRate
        elif self.updateRule == 'ifgsm':

            return input - grads.sign() * self.learningRate

    def distribution_matching_loss(self, pred, flabels):
        pred_mean = torch.mean(pred, dim=0)
        flabels_mean = torch.mean(flabels, dim=0)
        pred_std = torch.std(pred, dim=0)
        flabels_std = torch.std(flabels, dim=0)

        mean_loss = torch.mean((pred_mean - flabels_mean) ** 2)
        std_loss = torch.mean((pred_std - flabels_std) ** 2)

        combined_loss = mean_loss + std_loss
        return combined_loss

    def unspecificAttack(self, labels):
        flabels = np.ones((len(labels), self.class_num))
        flabels = flabels * 1 / self.class_num

        return torch.LongTensor(flabels)

    def attack(self):  # 执行攻击的主要方法，包括对训练数据进行迭代，计算分类损失和感知损失，更新对抗样本
        overallFoolRate = 0
        batchTotalNum = 0

        if os.path.exists(self.save_path) == False:
            os.makedirs(self.save_path)
        samples_x_list = []
        frames_list = []
        attck_samples_x_list = []
        attck_samples_y_list = []

        for batchNo, (tx, ty, tn) in enumerate(self.trainloader):
            print(f'batchNo={batchNo}')
            if self.auto_learn_ratio:
                self.resetAutoLearnRatio()
            tx = tx.cuda()

            # 根据攻击类型设置攻击标签
            labels = ty  # 进行攻击，把标签更改掉
            if self.attackType == 'mean':
                flabels = torch.ones((len(labels), self.class_num)).cuda()
                flabels = flabels * 1 / self.class_num
                for i in range(len(ty)):
                    flabels[i, ty[i]] = 0.001
            elif self.attackType == 'abn':
                flabels = self.unspecificAttack(labels).cuda()
            elif self.attackType == 'ab':
                flabels = labels.cuda()
            else:
                print('未设置attackType')
                return

            # 初始化一个空列表用于存储非异常数据
            valid_data = []
            valid_data_y = []
            valid_data_flabels = []

            # 迭代检查每一行
            for i in range(len(tx)):
                if torch.all(tx[i] == 0) == False:
                    valid_data.append(tx[i])
                    valid_data_y.append(ty[i])
                    valid_data_flabels.append(flabels[i])

            for i in range(len(tx) - len(valid_data)):
                valid_data.append(tx[0])
                valid_data_y.append(ty[0])
                valid_data_flabels.append(flabels[0])

            # 将非异常数据重新转换为张量
            tx = torch.stack(valid_data)
            ty = torch.stack(valid_data_y)
            flabels = torch.stack(valid_data_flabels)

            # 3维转2维，原始的2d样本
            tx_2d = ThreeDimensionsToTwoDimensions(tx)

            adData = tx.clone()  # 复制一份数据
            adData = adData.cuda()
            adData.requires_grad = True  # 设置为梯度可更新，样本数据可更新？更新adData的数据
            maxFoolRate = np.NINF
            batchTotalNum += 1
            IsFinished = False

            for ep in range(self.epochs):
                pred = self.classifier(adData)
                predictedLabels = torch.argmax(pred, axis=1)

                if self.attackType == 'cross':
                    classLoss = -torch.nn.CrossEntropyLoss()(pred, flabels)
                elif self.attackType == 'abn':
                    classLoss = torch.mean((pred - flabels) ** 2)
                elif self.attackType == 'mean':
                    classLoss = self.distribution_matching_loss(pred, flabels)
                else:
                    print('未设置attackType')
                    return

                adData.grad = None
                classLoss.backward(retain_graph=True)
                cgs = adData.grad

                adData_2d = ThreeDimensionsToTwoDimensions(adData)
                if self.perceptualType == 'projection':
                    # 二维投影损失
                    squaredLoss = K.sum(
                        K.reshape(K.square(tx_2d - adData_2d), (tx_2d.shape[0], tx_2d.shape[1], 25, -1)),
                        axis=-1)
                    squareCost = K.sum(K.sum(squaredLoss, axis=-1), axis=-1)
                    percepLoss = K.mean(squareCost, axis=-1)
                    pgs = percepLoss
                elif self.perceptualType == 'kinematics':
                    # 二维运动学损失
                    projection_loss = computer_perceptual_loss(tx_2d, adData_2d)
                    # 三维运动学损失
                    kinematics_loss = computer_perceptual_loss(tx, adData)
                    '''
                    投影到二维的话会带来坐标的损失，虽然三维运动学约束是会带来更小的约束，但是有这个三维运动学约束的存在会让梯度方向更高效
                    kinematics_loss系数越大意味着3d约束越强，二维约束越弱
                    '''
                    percepLoss = projection_loss + 0.01 * kinematics_loss

                    adData.grad = None
                    percepLoss.backward(retain_graph=True)
                    pgs = adData.grad

                    pgsView = pgs.view(pgs.shape[0], -1)
                    pgsnorms = torch.norm(pgsView, dim=1) + 1e-18
                    pgsView /= pgsnorms[:, np.newaxis]
                else:
                    print('未设置perceptualType参数')
                    return

                if ep % 50 == 0:
                    print(f"Iteration {ep}: Class Loss {classLoss:>9f}, pgs Loss: {percepLoss:>9f}")

                foolRate = self.foolRateCal(ty, predictedLabels)

                if maxFoolRate < foolRate:  # 记录针对于这个样本进行的所有epoch攻击中欺骗率最高的结果
                    print('foolRate Improved! Iteration %d, batchNo %d: Class Loss %.9f, pgs: %.9f, Fool rate:%.2f' % (
                        ep, batchNo, classLoss, percepLoss, foolRate))
                    maxFoolRate = foolRate

                if ep == self.epochs - 1 or IsFinished == True:  # 如果已经可以达到100%的欺骗率，则停止对这个样本进行攻击
                    for i in range(len(ty)):
                        if torch.all(tx[i] == 0):
                            print('这条原始数据异常了')
                        if torch.sum(tx[i] - adData[i]) != 0:
                            if ty[i] != predictedLabels[i]:  # 保证此条数据原本就是模型可以识别正确的
                                samples_x_list.append(tx[i].detach().clone().cpu())
                                frames_list.append(tn[i].detach().clone().cpu())
                                attck_samples_x_list.append(adData[i].detach().clone().cpu())
                                attck_samples_y_list.append(ty[i].detach().clone().cpu())
                    break

                cgsView = cgs.view(cgs.shape[0], -1)
                cgsnorms = torch.norm(cgsView, dim=1) + 1e-18
                cgsView /= cgsnorms[:, np.newaxis]

                temp = self.getUpdate(cgs * self.classWeight + pgs * (1 - self.classWeight), adData)

                if self.auto_learn_ratio: # 尽量是False，否则会最大化
                    self.learningRate = self.learningRate - (1 - cgs) * 0.0001  # 根据cgs大小调整学习率
                with torch.no_grad():
                    missedIndices = []
                    if self.perceptual_delta > 0:  # 取在扰动预算之内的
                        N, C, T, V, M = adData.shape
                        for i in range(len(adData)):
                            p = computer_pertual_2d_two(tx[i].view(1, C, T, V, M), adData[i].view(1, C, T, V, M))
                            if p < self.perceptual_delta:  # 小于感知阈值的样本才更新
                                missedIndices.append(i)
                    else:  # 取全部的数据
                        missedIndices = []
                        for i in range(len(ty)):
                            missedIndices.append(i)

                    if self.updateRule == 'gd':
                        updates = temp[missedIndices] - adData[missedIndices]
                        for ci in range(updates.shape[0]):
                            updateNorm = torch.norm(updates[ci])
                            if updateNorm > self.updateClip:
                                updates[ci] = updates[ci] * self.updateClip / updateNorm
                        adData[missedIndices] += updates
                    else:
                        adData[missedIndices] = temp[missedIndices]
                    adData.data = torch.clamp(adData.data, min=tx - self.clamp_delta, max=tx + self.clamp_delta)

                    if self.perceptual_delta > 0:
                        if len(missedIndices) == 0:  # 如果等于0说明没有需要更新的样本了，则直接停止
                            IsFinished = True

            overallFoolRate += maxFoolRate
            print(f"Current fool rate is {overallFoolRate / batchTotalNum}")

            if len(attck_samples_x_list) != 0:
                # 保存
                FileUtil.writer(samples_x_list, frames_list, attck_samples_x_list, attck_samples_y_list, self.save_path)

            if len(attck_samples_x_list) >= self.end_sample_num:
                print('攻击结束')
                break

        print(f"Overall fool rate is {overallFoolRate / batchTotalNum}")
        return overallFoolRate / batchTotalNum


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    attacker = Attacker(config)
    attacker.attack()
