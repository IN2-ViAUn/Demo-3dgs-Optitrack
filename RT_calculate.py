
import numpy as np
import json
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

def calculate_affine_matrix(source_points, target_points):
    ones = np.ones((source_points.shape[0], 1))
    source_points_augmented = np.hstack([source_points, ones])

    # 增加一个额外的0行以适配仿射变换的需要
    target_points_augmented = np.hstack([target_points, np.ones((target_points.shape[0], 1))])

    affine_matrix, residuals, rank, s = np.linalg.lstsq(source_points_augmented, target_points_augmented, rcond=None)
    return affine_matrix.T  # 转置以匹配期望的形状

pattern_size = (8, 5)  # 例如，8x5的棋盘格
pattern_length = 0.015 # 棋盘格子长度，单位m

def convert_quat_and_swap_euler(init_ori_cap_y):
    r1 = R.from_quat(init_ori_cap_y)            # xyzw 顺序的四元数
    init_euler_cap = r1.as_euler('xyz')         # 获取欧拉角 (roll, pitch, yaw)
    init_euler_cap[1], init_euler_cap[2] = init_euler_cap[2], init_euler_cap[1] # 交换 Y 和 Z 分量
    r2 = R.from_euler('xyz', init_euler_cap)    # 将修改后的欧拉角转换回四元数

    quat = r2.as_quat()                         # [x, y, z, w] 顺序的四元数
    return quat


# 场地坐标系下的点
motion_tracking_points = np.array([
    [ -0.6182,  0.4610, -0.8075],
    [ -0.2732,  0.4677,  0.0923],
    [  0.0482,  0.4517, -2.1868],
    [ -0.6753,  0.4404,  0.9429],
    [  1.2937,  0.4407,  0.0850],
    [  1.2194,  0.2206, -0.6788],
    [  1.0929,  0.6427, -1.8963],
    [ -1.1190,  0.1051,  0.4270],
    [ -1.1260,  0.358,  -1.346],
    [  0.59,  0.083,  0.255]
])

# 动捕坐标系下的点(交换了YZ)
camera_points = np.array([
    [-36.429,  1.910, 4.555 ],
    [-35.493,  1.873, 6.696 ],
    [-34.946,  1.943, 1.142 ],
    [-36.382,  1.758, 8.820 ],
    [-31.725,  1.704, 6.532 ],
    [-31.957,  1.180, 4.903 ],
    [-32.381,  2.315, 1.763 ],
    [-37.573,  0.929, 7.530 ],
    [-37.968,  1.690, 3.297 ],
    [-33.475,  0.873, 6.932 ]
])


print("camera_points:\n",camera_points)
# 计算仿射变换矩阵
affine_matrix = calculate_affine_matrix(motion_tracking_points, camera_points)
json_data = {
    "affine_matrix": affine_matrix.tolist()
}
with open("affine_matrix.json", "w") as file:
    json.dump(json_data, file)
print("affine_matrix:", affine_matrix)
# print("field after convert:", (affine_matrix @ motion_tracking_points.T).T)
# affine_matrix: [[-1.80878845e+00  1.84346208e+00 -1.24748448e+00  1.82101232e+00]
#  [-4.08563391e-01 -6.69964629e-01 -3.90635806e-02  4.55574551e-01]
#  [ 1.60496409e+00  1.34102302e+00 -1.25901454e+00  1.23909457e+00]
#  [ 9.98503943e-17  3.20763128e-16 -7.24159547e-16  1.00000000e+00]]

