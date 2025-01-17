import matplotlib.pyplot as plt
import imageio
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import torch

sk_adj = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24],
          [1, 20, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 22, 7, 24, 11]]


# 辅助固定空间轴
def addPoints(x, y, z):
    basex = 0.5
    basey = 0.5
    basez = 0.5

    x.append(basex)
    y.append(basey)
    z.append(basez)

    x.append(-basex)
    y.append(basey)
    z.append(basez)

    x.append(basex)
    y.append(-basey)
    z.append(basez)

    x.append(-basex)
    y.append(-basey)
    z.append(basez)

    x.append(basex)
    y.append(basey)
    z.append(-basez)

    x.append(-basex)
    y.append(basey)
    z.append(-basez)

    x.append(basex)
    y.append(-basey)
    z.append(-basez)

    x.append(-basex)
    y.append(-basey)
    z.append(-basez)


def draw(x_frames, y_frames, z_frames, x_frames2, y_frames2, z_frames2, save_folder, numframes, elev, azim):
    # 创建一个3D图形
    fig = plt.figure(figsize=(200, 10))
    axs = []
    for i in range(numframes * 2):
        axs.append(fig.add_subplot(2, numframes, i + 1, projection='3d'))

    # 添加链接线
    for f in range(numframes):
        # 绘制骨架线
        ax = axs[f]
        # addPoints(x_frames[f], y_frames[f], z_frames[f])
        ax.scatter(x_frames[f], y_frames[f], z_frames[f], s=5, c='#0070c0', marker='o', label='Skeleton Points')
        for i in range(len(sk_adj[0])):
            point1_index = sk_adj[0][i]
            point2_index = sk_adj[1][i]

            # 取出对应的坐标点
            x1, y1, z1 = x_frames[f][point1_index], y_frames[f][point1_index], z_frames[f][point1_index]
            x2, y2, z2 = x_frames[f][point2_index], y_frames[f][point2_index], z_frames[f][point2_index]

            # 绘制链接线
            ax.plot([x1, x2], [y1, y2], [z1, z2], color='#35b7f2', linestyle='-', linewidth=1)
            ax.set_xlim([-0.5, 0.5])
            ax.set_ylim([-0.5, 0.5])
            ax.set_zlim([-0.5, 0.5])

        # 添加标题和标签
        # ax.set_title('Original frame {}'.format(f))
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.view_init(elev, azim)
        # ax.axis('off')
        # ax.grid(False)
        # 显示图例
        # ax.legend()

    for f in range(numframes):
        # 绘制骨架线
        ax = axs[f + numframes]
        # addPoints(x_frames2[f], y_frames2[f], z_frames2[f])
        ax.scatter(x_frames2[f], y_frames2[f], z_frames2[f], s=5, c='#04b0f1', marker='o', label='Skeleton Points2')
        for i in range(len(sk_adj[0])):
            point1_index = sk_adj[0][i]
            point2_index = sk_adj[1][i]

            # 取出对应的坐标点
            x1, y1, z1 = x_frames2[f][point1_index], y_frames2[f][point1_index], z_frames2[f][point1_index]
            x2, y2, z2 = x_frames2[f][point2_index], y_frames2[f][point2_index], z_frames2[f][point2_index]

            # 绘制链接线
            ax.plot([x1, x2], [y1, y2], [z1, z2], color='#488fce', linestyle='-', linewidth=1)
            ax.set_xlim([-0.5, 0.5])
            ax.set_ylim([-0.5, 0.5])
            ax.set_zlim([-0.5, 0.5])

        # 添加标题和标签
        # ax.set_title('Perturb frame {}'.format(f))
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.view_init(elev, azim)
        # ax.axis('off')
        # ax.grid(False)
        # 显示图例
        # ax.legend()

    # 显示图形
    plt.savefig('{}'.format(save_folder), format="svg")


# 叠加两幅骨架图在一起
def draw2(x_frames, y_frames, z_frames, x_frames2, y_frames2, z_frames2, save_folder, numframes, elev, azim):
    # 创建一个3D图形
    fig = plt.figure(figsize=(300, 10))
    axs = []
    for i in range(numframes):
        axs.append(fig.add_subplot(1, numframes, i + 1, projection='3d'))

    # 添加链接线
    for f in range(numframes):
        # 绘制骨架线
        ax = axs[f]
        ax.scatter(x_frames[f], y_frames[f], z_frames[f], s=1, c='b', marker='o', label='Skeleton Points')
        ax.scatter(x_frames2[f], y_frames2[f], z_frames2[f], s=1, c='#A020F0', marker='o', label='Skeleton Points')
        for i in range(len(sk_adj[0])):
            point1_index = sk_adj[0][i]
            point2_index = sk_adj[1][i]

            # 取出对应的坐标点
            x1, y1, z1 = x_frames[f][point1_index], y_frames[f][point1_index], z_frames[f][point1_index]
            x2, y2, z2 = x_frames[f][point2_index], y_frames[f][point2_index], z_frames[f][point2_index]
            ax.plot([x1, x2], [y1, y2], [z1, z2], color='r', linestyle='-', linewidth=1)

            x1, y1, z1 = x_frames2[f][point1_index], y_frames2[f][point1_index], z_frames2[f][point1_index]
            x2, y2, z2 = x_frames2[f][point2_index], y_frames2[f][point2_index], z_frames2[f][point2_index]
            ax.plot([x1, x2], [y1, y2], [z1, z2], color='g', linestyle='-', linewidth=1)
            ax.set_xlim([-0.5, 0.5])
            ax.set_ylim([-0.5, 0.5])
            ax.set_zlim([-0.5, 0.5])

        # 添加标题和标签
        ax.set_title('Original frame {}'.format(f))
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.view_init(elev, azim)
        # ax.axis('off')
        # ax.grid(False)
        # 显示图例
        # ax.legend()

    # 显示图形
    plt.savefig('{}'.format(save_folder), format="svg")


def create_gif(data, filename='./img/skeleton_animation.gif'):
    C, T, V = data.shape
    assert C == 3, "The first dimension of data must be 3 for x, y, z coordinates."

    # 创建一个列表来存储每一帧的图像
    frames = []

    # 创建每一帧的图像
    for t in range(T):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        x = data[0, t, :]
        y = data[1, t, :]
        z = data[2, t, :]

        # 绘制骨骼点
        ax.scatter(x, y, z, color='blue')

        # 绘制骨骼连接线
        for i in range(len(sk_adj[0])):
            point1_index = sk_adj[0][i]
            point2_index = sk_adj[1][i]
            x1, y1, z1 = x[point1_index], y[point1_index], z[point1_index]
            x2, y2, z2 = x[point2_index], y[point2_index], z[point2_index]
            ax.plot([x1, x2], [y1, y2], [z1, z2], color='red')

        # 将当前帧的图像保存为内存中的一个文件对象
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close(fig)

    # 使用 imageio 将帧序列保存为 GIF 动图
    imageio.mimsave(filename, frames, fps=10)


# 将原图64张画到一张上面
def drawSkeletonByIndex(data, pedata, save_folder, type=0, elev=30, azim=45):
    # CTVM
    C, T, V, M = data.shape
    data = data.permute(1, 3, 2, 0)  # TMVC
    pedata = pedata.permute(1, 3, 2, 0)

    x_frames = []
    y_frames = []
    z_frames = []
    x_frames2 = []
    y_frames2 = []
    z_frames2 = []
    for t in range(T):
        x_points = []
        y_points = []
        z_points = []
        x_points2 = []
        y_points2 = []
        z_points2 = []
        for j in range(25):
            x_points.append(data[t][0][j][0].detach().cpu())
            y_points.append(data[t][0][j][1].detach().cpu())
            z_points.append(data[t][0][j][2].detach().cpu())
            x_points2.append(pedata[t][0][j][0].detach().cpu())
            y_points2.append(pedata[t][0][j][1].detach().cpu())
            z_points2.append(pedata[t][0][j][2].detach().cpu())
        x_frames.append(x_points)
        y_frames.append(y_points)
        z_frames.append(z_points)
        x_frames2.append(x_points2)
        y_frames2.append(y_points2)
        z_frames2.append(z_points2)


    if T > 100:
        T = 100

    if type == 0:
        draw(x_frames, y_frames, z_frames, x_frames2, y_frames2, z_frames2, save_folder, T, elev, azim)
    else:
        draw2(x_frames, y_frames, z_frames, x_frames2, y_frames2, z_frames2, save_folder, T, elev, azim)


def drawSkeletonByIndex2(data, pedata, save_folder, type=0, elev=35, azim=45):
    frames = []
    C, T, V, M = data.shape
    for t in range(T):
        for v in range(V):
            for m in range(M):
                diff = data[:, t, v, m] - pedata[:, t, v, m]
                print(diff)
                # if diff > 0.0173:
                #     frames.append(t)

    # CTVM
    data = data.permute(1, 3, 2, 0)  # TMVC
    pedata = pedata.permute(1, 3, 2, 0)

    x_frames = []
    y_frames = []
    z_frames = []
    x_frames2 = []
    y_frames2 = []
    z_frames2 = []
    T = len(frames)
    for f in range(T):
        t = frames[f]
        x_points = []
        y_points = []
        z_points = []
        x_points2 = []
        y_points2 = []
        z_points2 = []
        for j in range(25):
            x_points.append(data[t][0][j][0].detach().cpu())
            y_points.append(data[t][0][j][1].detach().cpu())
            z_points.append(data[t][0][j][2].detach().cpu())
            x_points2.append(pedata[t][0][j][0].detach().cpu())
            y_points2.append(pedata[t][0][j][1].detach().cpu())
            z_points2.append(pedata[t][0][j][2].detach().cpu())
        x_frames.append(x_points)
        y_frames.append(y_points)
        z_frames.append(z_points)
        x_frames2.append(x_points2)
        y_frames2.append(y_points2)
        z_frames2.append(z_points2)

    if T > 100:
        T = 100

    if type == 0:
        draw(x_frames, y_frames, z_frames, x_frames2, y_frames2, z_frames2, save_folder, T, elev, azim)
    else:
        draw2(x_frames, y_frames, z_frames, x_frames2, y_frames2, z_frames2, save_folder, T, elev, azim)


# 画扰动图
def Plotting_perturbations(data, save_folder):
    # 定义自定义颜色映射
    colors = [(0, "#8b89c0"), (0.2, "#d4cee6"), (0.4, "#d4cee6"), (0.6, "#f4e77a"), (0.9, "#ef8469"),
              (1.0, "#ffffff")]  # 从蓝色到绿色到红色
    n_bins = 256  # 颜色的数量
    cmap_name = 'custom_cmap'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    data = data[:, :, :, :, 0]
    N, C, T, V = data.shape
    data = data.permute(0, 2, 1, 3)
    data = data.reshape(N * T, C, 5, 5).cpu()  # NCHW

    # weights = np.array([0.2989, 0.5870, 0.1140])  # 对应于RGB图像的标准灰度转换权重

    # 检查数据的通道数是否和权重匹配
    # if data.shape[1] == len(weights):
    #     gray_data = np.tensordot(data, weights, axes=([1], [0]))
    #     print(gray_data.shape)  # 输出 (N, H, W)
    # else:
    #     raise ValueError("数据的通道数和权重长度不匹配")

    # N, H, W = gray_data.shape

    # 按照C维度将数值相加
    sum_data = torch.sum(data, axis=1)  # 结果是一个形状为 (N, H, W) 的数组
    # 除以3
    average_data = sum_data / 3.0
    average_data = average_data * 10
    N = 64
    # 创建一个图形窗口
    fig = plt.figure(figsize=(200, 10))
    axs = []
    for i in range(N):
        axs.append(fig.add_subplot(1, N, i + 1))

    # 遍历每个样本并显示图像
    for i in range(N):
        axs[i].imshow(average_data[i], cmap=custom_cmap)
        axs[i].set_title(f'Sample {i + 1}')
        axs[i].axis('off')

    plt.savefig('{}'.format(save_folder), format="svg")


# 查找数据中扰动最大帧进行绘制
def MaxDrawFrames(data, pedata, delta, save_folder, type=0, elev=30, azim=45):
    data = data.cpu()
    pedata = pedata.cpu()
    C, T, V, M = data.shape
    tx_temp = []
    x_temp = []
    for t in range(T):
        for m in range(M):
            if torch.max(torch.abs(data[:, t, :, m] - pedata[:, t, :, m])) > delta:
                tx_temp.append(data[:, t, :, m])
                x_temp.append(pedata[:, t, :, m])
    if len(tx_temp) == 0:
        print(f'无扰动超过{delta}')
        return
    tx_temp = torch.tensor(np.stack(tx_temp))
    tx_temp = tx_temp.permute(1, 0, 2)
    tx_temp = tx_temp.reshape(tx_temp.shape[0], tx_temp.shape[1], tx_temp.shape[2], 1)
    x_temp = torch.tensor(np.stack(x_temp))
    print(f'有{len(x_temp)}的扰动超过了{delta}')
    x_temp = x_temp.permute(1, 0, 2)
    x_temp = x_temp.reshape(x_temp.shape[0], x_temp.shape[1], x_temp.shape[2], 1)
    drawSkeletonByIndex(tx_temp, x_temp, save_folder, type, elev, azim)
