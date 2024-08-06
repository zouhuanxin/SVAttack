import os

from utils.loadModel import *
from utils.ViewUtil import *
from utils.MiddleUtils import *
from torch.utils.data import DataLoader
from feeder.feeder import *
import torch
import torch as K
import argparse
from omegaconf import OmegaConf


class Attacker():
    def __init__(self, args):
        super().__init__()
        self.name = 'SingleViewAttack'
        print(f'运行参数 classWeight={args.classWeight} epochs={args.epochs} updateClip={args.updateClip}'
              f' model_name={args.model_name} hookstatus={args.hookstatus}')
        self.classWeight = args.classWeight
        self.epochs = args.epochs
        self.updateClip = args.updateClip
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.classifier = getModel(self.model_name)

        data_path = args.data_path
        label_path = args.label_path
        num_frame_path = args.num_frame_path
        feeder = Feeder(data_path, label_path, num_frame_path)
        self.trainloader = DataLoader(feeder,
                                      batch_size=args.batch_size,
                                      num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
        if args.hookstatus:
            register_hooks(self.classifier)

    def foolRateCal(self, rlabels, flabels):  # 计算欺骗率，即成功攻击的样本所占比例
        hitIndices = []

        for i in range(0, len(flabels)):
            if flabels[i] != rlabels[i]:
                hitIndices.append(i)

        return len(hitIndices) / len(flabels) * 100

    def getUpdate(self, grads, input):  # 获取更新后的对抗样本
        self.learningRate = 0.01

        return input - grads * self.learningRate

    def reshapeData(self, x, toNative=True):  # 调整数据格式
        if toNative:
            x = x.permute(0, 2, 3, 1, 4)
            x = x.reshape((x.shape[0], x.shape[1], -1, x.shape[4]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], -1, 3, x.shape[4]))
            x = x.permute(0, 3, 1, 2, 4)
        return x

    def distribution_matching_loss(self, pred, flabels):
        pred_mean = torch.mean(pred, dim=0)
        flabels_mean = torch.mean(flabels, dim=0)
        pred_std = torch.std(pred, dim=0)
        flabels_std = torch.std(flabels, dim=0)

        mean_loss = torch.mean((pred_mean - flabels_mean) ** 2)
        std_loss = torch.mean((pred_std - flabels_std) ** 2)

        combined_loss = mean_loss + std_loss
        return combined_loss

    def attack(self):  # 执行攻击的主要方法，包括对训练数据进行迭代，计算分类损失和感知损失，更新对抗样本
        global middle_flabels
        overallFoolRate = 0
        batchTotalNum = 0

        if os.path.exists(f'{self.save_path}/source_x_list.npy'):
            source_x_list = np.load(f'{self.save_path}/source_x_list.npy')
            frames_list = np.load(f'{self.save_path}/frames_list.npy')
            attck_x_list = np.load(f'{self.save_path}/attck_x_list.npy')
            source_y_list = np.load(f'{self.save_path}/source_y_list.npy')
            source_x_list = source_x_list.tolist()
            frames_list = frames_list.tolist()
            attck_x_list = attck_x_list.tolist()
            source_y_list = source_y_list.tolist()
        else:
            if os.path.exists(self.save_path) == False:
                os.mkdir(self.save_path)
            source_x_list = []
            frames_list = []
            attck_x_list = []
            source_y_list = []

        for batchNo, (tx, ty, tn) in enumerate(self.trainloader):
            print(f'batchNo={batchNo}')
            tx = tx.cuda()

            labels = ty
            flabels = torch.ones((len(labels), 60)).cuda()
            flabels = flabels * 1 / 60

            valid_data = []
            valid_data_y = []
            valid_data_flabels = []

            # 迭代检查每一行
            for i in range(len(tx)):
                if torch.all(tx[i] == 0) == False:
                    valid_data.append(tx[i])
                    valid_data_y.append(ty[i])
                    valid_data_flabels.append(flabels[i])

            tx = torch.stack(valid_data)
            ty = torch.stack(valid_data_y)
            flabels = torch.stack(valid_data_flabels)

            tx_2d = ThreeDimensionsToTwoDimensions(tx)

            adData = tx.clone()  # 复制一份数据
            adData = adData.cuda()
            adData.requires_grad = True
            maxFoolRate = np.NINF
            batchTotalNum += 1

            for ep in range(self.epochs):
                pred = self.classifier(adData)
                predictedLabels = torch.argmax(pred, axis=1)

                classLoss = self.distribution_matching_loss(pred, flabels)

                adData.grad = None
                classLoss.backward(retain_graph=True)
                cgs = adData.grad

                adData_2d = ThreeDimensionsToTwoDimensions(adData)
                squaredLoss = K.sum(K.reshape(K.square(tx_2d - adData_2d), (tx_2d.shape[0], tx_2d.shape[1], 25, -1)),
                                    axis=-1)
                squareCost = K.sum(K.sum(squaredLoss, axis=-1), axis=-1)
                vgs = K.mean(squareCost, axis=-1)

                if ep % 50 == 0:
                    print(f"Iteration {ep}: Class Loss {classLoss:>9f}, vgs Loss: {vgs:>9f}")

                foolRate = self.foolRateCal(ty, predictedLabels)

                if maxFoolRate < foolRate:
                    print('foolRate Improved! Iteration %d, batchNo %d: Class Loss %.9f, vgs: %.9f, Fool rate:%.2f' % (
                        ep, batchNo, classLoss, vgs, foolRate))
                    maxFoolRate = foolRate

                if ep == self.epochs - 1:
                    for i in range(len(ty)):
                        if torch.sum(tx[i] - adData[i]) != 0:
                            if ty[i] != predictedLabels[i]:
                                source_x_list.append(tx[i].detach().clone().cpu())
                                frames_list.append(tn[i].detach().clone().cpu())
                                attck_x_list.append(adData[i].detach().clone().cpu())
                                source_y_list.append(ty[i].detach().clone().cpu())
                    break

                cgsView = cgs.view(cgs.shape[0], -1)

                cgsnorms = torch.norm(cgsView, dim=1) + 1e-18

                cgsView /= cgsnorms[:, np.newaxis]

                with torch.no_grad():
                    temp = self.getUpdate(cgs * self.classWeight + vgs * (1 - self.classWeight), adData)
                    missedIndices = []
                    for i in range(len(ty)):
                        missedIndices.append(i)

                    if self.updateClip > 0:
                        updates = temp[missedIndices] - adData[missedIndices]
                        for ci in range(updates.shape[0]):
                            updateNorm = torch.norm(updates[ci])
                            if updateNorm > self.updateClip:
                                updates[ci] = updates[ci] * self.updateClip / updateNorm

                        adData[missedIndices] += updates
                    else:
                        adData[missedIndices] = temp[missedIndices]

            overallFoolRate += maxFoolRate
            print(f"Current fool rate is {overallFoolRate / batchTotalNum}")

            if len(attck_x_list) != 0:
                # 保存
                np.save(f'{self.save_path}/source_x_list.npy', np.stack(source_x_list))
                np.save(f'{self.save_path}/frames_list.npy', np.stack(frames_list))
                np.save(f'{self.save_path}/attck_x_list.npy', np.stack(attck_x_list))
                np.save(f'{self.save_path}/source_y_list.npy', np.stack(source_y_list))
                print(f'当前样本数量={len(attck_x_list)} 当前batchNo={batchNo}')

        print(f"Overall fool rate is {overallFoolRate / batchTotalNum}")
        return overallFoolRate / batchTotalNum
''''''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/stgcn.yaml")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    attacker = Attacker(config)
    attacker.attack()

