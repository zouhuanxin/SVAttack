import numpy as np
import torch


def look_at(eye, target, up):
    eye = torch.tensor(eye, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)
    up = torch.tensor(up, dtype=torch.float32)

    forward = target - eye
    forward = forward / torch.norm(forward)

    right = torch.cross(forward, up)
    right = right / torch.norm(right)

    up = torch.cross(right, forward)

    view_matrix = torch.stack([
        torch.cat([right, torch.tensor([-torch.dot(right, eye)])]),
        torch.cat([up, torch.tensor([-torch.dot(up, eye)])]),
        torch.cat([-forward, torch.tensor([torch.dot(forward, eye)])]),
        torch.tensor([0, 0, 0, 1], dtype=torch.float32)
    ], dim=0)

    return view_matrix


def perspective(fov, aspect, near, far):
    f = 1.0 / torch.tan(fov / 2.0)
    depth = near - far

    proj_matrix = torch.tensor([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / depth, (2 * far * near) / depth],
        [0, 0, -1, 0]
    ], dtype=torch.float32)

    return proj_matrix


def project(points, view_matrix, proj_matrix):
    ones = torch.ones((*points.shape[:-1], 1), dtype=points.dtype, device=points.device)
    points = torch.cat((points, ones), dim=-1)
    view_points = torch.matmul(points, view_matrix.t())
    proj_points = torch.matmul(view_points, proj_matrix.t())
    proj_points = proj_points / proj_points[..., 3].unsqueeze(-1)
    return proj_points[..., :2]


'''
根据设置的仰角和方位角计算出相机的位置
'''


def compute_angle(angle_elevation=30, angle_azimuth=-60):
    # 定义新的仰角和方位角（以度为单位）
    elev_deg = angle_elevation
    azim_deg = angle_azimuth

    # 转换为弧度
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)

    # 计算距离 r（保持与原来相同）
    r = np.sqrt(3.0 ** 2 + 3.0 ** 2 + 3.0 ** 2)

    # 计算新的相机位置
    x_new = r * np.cos(elev) * np.cos(azim)
    y_new = r * np.cos(elev) * np.sin(azim)
    z_new = r * np.sin(elev)

    eye_new = np.array([x_new, y_new, z_new])

    return eye_new


def ThreeDimensionsToTwoDimensions(data, elev=None, azim=None):
    device = data.device
    N, C, T, V, M = data.shape

    # 视角参数
    if elev != None:
        eye = compute_angle(elev, azim)
    else:
        eye = [3.0, 3.0, 3.0]
    target = [0.0, 0.0, 0.0]
    up = [0.0, 1.0, 0.0]
    # 投影参数
    fov = torch.deg2rad(torch.tensor(90.0))
    aspect = 1.0
    near = 0.1
    far = 100.0

    # 计算视图矩阵
    view_matrix = look_at(eye, target, up)
    view_matrix = view_matrix.to(device)
    # 计算投影矩阵
    proj_matrix = perspective(fov, aspect, near, far)
    proj_matrix = proj_matrix.to(device)

    # 处理数据并进行投影
    data = data.permute(0, 2, 3, 4, 1).contiguous()  # [N, T, V, M, C]
    data_reshaped = data.view(-1, C)  # 展平到 [N*T*V*M, C]
    points_2d = project(data_reshaped, view_matrix, proj_matrix)
    output_data = points_2d.view(N, T, V, M, 2)  # 还原到 [N, T, V, M, 2]
    output_data = output_data.permute(0, 4, 1, 2, 3)  # [N, 2, T, V, M]

    return output_data
