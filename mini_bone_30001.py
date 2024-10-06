import sys
import os
import numpy as np
import time
import torch
import math
import network_gui
import socket
import threading
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation
from G4DGS.G4DGS import G4DGS
from render import simple_render
from queue import Queue
import curses

MAX_CLIENTS = 2
BUFFER_SIZE = 1024
RECEIVE_PORT = 30001
cout_mutex = threading.Lock()       # 互斥锁，用于多线程安全地输出
message_queue = Queue()

# 小车初始位置在动捕坐标系下的信息：
init_pos_cap = np.array([-0.3517, 0.0834, 0.4709])              # 这里已经交换了 YZ
init_qua_cap_y = np.array([ 0.0106, -0.0425, -0.0346, -0.9984])   # xyzw 顺序
cap2fld = torch.tensor([ 
    [-1.80878845e+00,  1.84346208e+00, -1.24748448e+00,  1.82101232e+00],
    [-4.08563391e-01, -6.69964629e-01, -3.90635806e-02,  4.55574551e-01],
    [ 1.60496409e+00,  1.34102302e+00, -1.25901454e+00,  1.23909457e+00],
    [ 9.98503943e-17,  3.20763128e-16, -7.24159547e-16,  1.00000000e+00]], 
    dtype=torch.float, device="cuda").unsqueeze(0)

cap2fld_R = np.array([
    [-1.80878845e+00,  1.84346208e+00, -1.24748448e+00],
    [-4.08563391e-01, -6.69964629e-01, -3.90635806e-02],
    [ 1.60496409e+00,  1.34102302e+00, -1.25901454e+00]
])

# 单位四元数转化为旋转矩阵，注意此时转换的四元数顺序为：wxyz
def unitquat_to_rotmat(quat):
    
    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    matrix = torch.empty(quat.shape[:-1] + (3, 3), dtype=quat.dtype, device=quat.device)

    matrix[..., 0, 0] = x2 - y2 - z2 + w2
    matrix[..., 1, 0] = 2 * (xy + zw)
    matrix[..., 2, 0] = 2 * (xz - yw)

    matrix[..., 0, 1] = 2 * (xy - zw)
    matrix[..., 1, 1] = - x2 + y2 - z2 + w2
    matrix[..., 2, 1] = 2 * (yz + xw)

    matrix[..., 0, 2] = 2 * (xz + yw)
    matrix[..., 1, 2] = 2 * (yz - xw)
    matrix[..., 2, 2] = - x2 - y2 + z2 + w2
    
    return matrix

def unflatten_batch_dims(tensor, batch_shape):
    return tensor.reshape(batch_shape + tensor.shape[1:]) if len(batch_shape) > 0 else tensor.squeeze(0)

def flatten_batch_dims(tensor, end_dim):
    batch_shape = tensor.shape[:end_dim+1]
    flattened = tensor.flatten(end_dim=end_dim) if len(batch_shape) > 0 else tensor.unsqueeze(0)
    return flattened, batch_shape

# 旋转矩阵转化为单位四元数，注意此时转换的四元数顺序为：wxyz
def rotmat_to_unitquat(R):

    matrix, batch_shape = flatten_batch_dims(R, end_dim=-3)
    num_rotations, D1, D2 = matrix.shape
    assert((D1, D2) == (3,3)), "Input should be a Bx3x3 tensor."

    decision_matrix = torch.empty((num_rotations, 4), dtype=matrix.dtype, device=matrix.device)
    decision_matrix[:, :3] = matrix.diagonal(dim1=1, dim2=2)
    decision_matrix[:, -1] = decision_matrix[:, :3].sum(axis=1)
    choices = decision_matrix.argmax(axis=1)

    quat = torch.empty((num_rotations, 4), dtype=matrix.dtype, device=matrix.device)

    ind = torch.nonzero(choices != 3, as_tuple=True)[0]
    i = choices[ind]
    j = (i + 1) % 3
    k = (j + 1) % 3

    quat[ind, i] = 1 - decision_matrix[ind, -1] + 2 * matrix[ind, i, i]
    quat[ind, j] = matrix[ind, j, i] + matrix[ind, i, j]
    quat[ind, k] = matrix[ind, k, i] + matrix[ind, i, k]
    quat[ind, 3] = matrix[ind, k, j] - matrix[ind, j, k]

    ind = torch.nonzero(choices == 3, as_tuple=True)[0]
    quat[ind, 0] = matrix[ind, 2, 1] - matrix[ind, 1, 2]
    quat[ind, 1] = matrix[ind, 0, 2] - matrix[ind, 2, 0]
    quat[ind, 2] = matrix[ind, 1, 0] - matrix[ind, 0, 1]
    quat[ind, 3] = 1 + decision_matrix[ind, -1]

    quat = quat / torch.norm(quat, dim=1)[:, None]
    quat = quat[:, [3, 0, 1, 2]]        # 改变顺序为 wxyz
    return unflatten_batch_dims(quat, batch_shape)

# 用于调整动捕坐标系的 YZ 四元数坐标，输出的四元数是 wxyz 顺序，格式均为 np.array
def convert_quat_and_swap_euler(init_ori_cap_y):
    r1 = Rotation.from_quat(init_ori_cap_y)     # xyzw 顺序的四元数
    init_euler_cap = r1.as_euler('xyz')         # 获取欧拉角 (roll, pitch, yaw)
    init_euler_cap[1], init_euler_cap[2] = init_euler_cap[2], init_euler_cap[1] # 交换 Y 和 Z 分量
    r2 = Rotation.from_euler('xyz', init_euler_cap)     # 将修改后的欧拉角转换回四元数

    quat = r2.as_quat()                         # [x, y, z, w] 顺序的四元数
    return [quat[3], quat[0], quat[1], quat[2]] # 调整为 [w, x, y, z] 顺序

def on_V_key_press(self):
    center = torch.mean(self.mean3D, dim=0)
    # 旋转矩阵
    angle = 180
    R = torch.tensor([[1, 0, 0], [0, math.cos(angle*math.pi/180), -math.sin(angle*math.pi/180)], [0, math.sin(angle*math.pi/180), math.cos(angle*math.pi/180)]]).cuda().float()
    with torch.no_grad():
        self.mean3D = torch.matmul(R, self.mean3D.T).T
        
        tmp_rotation = self.rots                    # 对四元数重新排列，由wxyz转化为xyzw
        rotation = torch.zeros_like(tmp_rotation)   
        rotation[..., 0] = tmp_rotation[..., 1]     # 把 x 位置的数赋给 rotation[..., 0]
        rotation[..., 1] = tmp_rotation[..., 2]     # 把 y 位置的数赋给 rotation[..., 1]
        rotation[..., 2] = tmp_rotation[..., 3]     # 把 z 位置的数赋给 rotation[..., 2]
        rotation[..., 3] = tmp_rotation[..., 0]     # 把 w 位置的数赋给 rotation[..., 3]

        cur_rot = unitquat_to_rotmat(rotation)
        rot_mat = R.unsqueeze(0) 
        new_rot = torch.matmul(rot_mat, cur_rot)
        new_quat = rotmat_to_unitquat(new_rot)

        self.rots[..., 0] = new_quat[..., 3]        # 把 xyzw 顺序的四元数以 wxyz 的顺序赋值给 self.rots
        self.rots[..., 1] = new_quat[..., 0]
        self.rots[..., 2] = new_quat[..., 1]
        self.rots[..., 3] = new_quat[..., 2]
    return True

# 处理客户端连接
def handle_client(client_socket, client_id):
    while True:
        try:
            buffer = client_socket.recv(BUFFER_SIZE).decode('utf-8')    # 接收客户端消息
            if buffer:
                if len(buffer.split()) >= 8:
                    message_queue.put(buffer.split()[:8])         # 只把前八位有用数据放入buffer
                else:
                    print(f"Incomplete data received: {buffer.split()}")# 数据不足8个时，跳过并打印警告
            else:
                print(f"Client {client_id} disconnected.")
                break
        except Exception as e:
            print(f"Error: {e}")
            break

    client_socket.close()

def handle_client_connections():
    while True:
        for client_id in range(1, MAX_CLIENTS + 1):
            client_socket, addr = server_socket.accept()
            print(f"Client {client_id} connected. IP: {addr[0]}")

            client_thread = threading.Thread(target=handle_client, args=(client_socket, client_id))
            client_thread.start()

def process_queue():
    while True:
        if not message_queue.empty():
            buffer = message_queue.get()
            print(f"Main thread received: {buffer}")
        else:
            print("message_queue is empty\n")
        # 加入一些延迟以避免不断轮询
        time.sleep(1)

# 将旋转矩阵和平移矩阵合并为 RT
def create_rt_matrix(rotation_matrix, translation_vector):
    if rotation_matrix.shape != (3, 3):     # 检查输入的形状
        raise ValueError("Rotation matrix must be 3x3.")
    if translation_vector.shape != (3, 1) and translation_vector.shape != (3,):
        raise ValueError("Translation vector must be 3x1 or 3x.")

    # 如果平移向量是1D数组，则将其转换为列向量
    if translation_vector.ndim == 1:
        translation_vector = translation_vector[:, np.newaxis]

    # 创建一个4x4的RT矩阵
    rt_matrix = np.eye(4)  # 初始化为单位矩阵
    rt_matrix[:3, :3] = rotation_matrix  # 将旋转矩阵填入左上角
    rt_matrix[:3, 3] = translation_vector.flatten()  # 将平移向量填入最后一列

    return rt_matrix

def rotate_o(gs2, R):
    # print("RRRRR", R)
    center_car = gs2.mean3D.mean(dim=0)
    # print("center_car", center_car)
    after_T_car = gs2.mean3D - center_car
    rotated = after_T_car @ R.T


    cur_rot = unitquat_to_rotmat(gs2.rots)
    rot_mat = R.T.unsqueeze(0)
    new_rot = torch.matmul(rot_mat, cur_rot)
    
    return rotated + center_car, rotmat_to_unitquat(new_rot).squeeze(0)

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    将欧拉角（roll, pitch, yaw）转换为 3x3 旋转矩阵。
    
    参数:
    roll: 绕 X 轴旋转（弧度）
    pitch: 绕 Y 轴旋转（弧度）
    yaw: 绕 Z 轴旋转（弧度）
    
    返回:
    3x3 旋转矩阵
    """
    
    # 绕 X 轴的旋转矩阵
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    # 绕 Y 轴的旋转矩阵
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    # 绕 Z 轴的旋转矩阵
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    # 组合旋转矩阵 R = Rz * Ry * Rx
    R = np.dot(R_z, np.dot(R_y, R_x))
    
    return R

my_xx = 0.0
my_yy = 0.0
my_zz = 0.0

def main(stdscr):
    global my_xx, my_yy, my_zz, server_socket
    stdscr.clear()
    stdscr.nodelay(True)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', RECEIVE_PORT))
    server_socket.listen(MAX_CLIENTS)

    print(f"Server listening on port {RECEIVE_PORT}...")

    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--workspace', type=str, default="/home/wangyixian/yXe_file/3DGS/mini_gs/data_xe")
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6010)
    parser.add_argument('--path', type=str, default="/home/wangyixian/yXe_file/3DGS/gaussian-splatting/data/lego")
    args = parser.parse_args(sys.argv[1:])

    path = args.path
    ip = args.ip
    port = args.port
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建背景并初始化
    gaussians_scene =  G4DGS(3,3)
    gaussians_scene.init_scene_from_plyfile(os.path.join(args.workspace, 'field_zheng.ply'),True)
    gs2 = gaussians_scene._load_ply(os.path.join(args.workspace, 'car_zheng.ply'),True)
    print(f"Initializing network GUI on IP: {ip}, Port: {port}")
    network_gui.init(ip, port)

    # 经过这些操作后，
    gs2.mean3D *= 0.15
    gs2.scales *= 0.15
    gs2.mean3D[:, 0] += 2    # x 坐标
    gs2.mean3D[:, 1] += 0.23  # y 坐标
    gs2.mean3D[:, 2] += 0  # z 坐标

    init_qua_cap_z = convert_quat_and_swap_euler(init_qua_cap_y)    # 改变初始位置在动捕坐标系下的 YZ，输出四元数为 wxyz
    init_rot_cap = unitquat_to_rotmat(torch.from_numpy(np.array(init_qua_cap_z))) # 输出为旋转矩阵
    init_RT = create_rt_matrix(init_rot_cap, init_pos_cap)

    connection_thread = threading.Thread(target=handle_client_connections)
    connection_thread.start()
    while True:
        if not message_queue.empty():
            buffer = message_queue.get()
            # print(f"Main thread received: {buffer}")
        else:
            buffer = ['1', '1', '1', '1', '1', '1', '1', '1']
            # print("Main thread is empty")
            
        id, x, y, z, ox, oy, oz, ow = map(float, buffer)    
        now_pos_cap = np.array([x,z,y])       # 注意这里交换了 YZ
        now_qua_cap_z = convert_quat_and_swap_euler([ox,oy,oz,ow])          # 转换四元数的坐标轴

        now_rot_cap = unitquat_to_rotmat(torch.from_numpy(np.array(now_qua_cap_z)))   # wxyz 顺序的四元数转旋转矩阵
        
        

        now_RT = create_rt_matrix(init_rot_cap, now_pos_cap)                # 当前小车的RT矩阵（动捕坐标系下）

        one = torch.ones((1, 1), device="cuda")
        homo = torch.from_numpy(np.array([now_pos_cap[0], now_pos_cap[1], now_pos_cap[2],1])).to("cuda")
        now_pos_fld = (cap2fld.float() @ homo.float().T).T[:, :3].squeeze(-1)
        
        
        # connecting SIBR Veiwer
        if network_gui.conn == None:
            # print('Try to connect...')
            network_gui.try_connect()
        while network_gui.conn != None:
            net_image_bytes = None
            custom_cam, _, _, _, keep_alive, scaling_modifer = network_gui.receive()
            # print("custom_cam:",custom_cam)
            if custom_cam != None:
                # 也可以参考用下面这部分代码代替，这部分代码是原版
                # net_image = simple_render(custom_cam, gaussians_scene.get_gaussians(), background)
                # net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                steppp = 0.1
                key = stdscr.getch()
                if key == ord('q'):
                    my_xx += steppp
                    print(f"my_xx += {steppp}, now = ({my_xx:.3f}, {my_yy:.3f}, {my_zz:.3f})\n")
                elif key == ord('w'):
                    my_xx -= steppp
                    print(f"my_xx -= {steppp}, now = ({my_xx:.3f}, {my_yy:.3f}, {my_zz:.3f})\n")
                elif key == ord('e'):
                    my_yy += steppp
                    print(f"my_yy += {steppp}, now = ({my_xx:.3f}, {my_yy:.3f}, {my_zz:.3f})\n")
                elif key == ord('r'):
                    my_yy -= steppp
                    print(f"my_yy -= {steppp}, now = ({my_xx:.3f}, {my_yy:.3f}, {my_zz:.3f})\n")
                elif key == ord('t'):
                    my_zz += steppp
                    print(f"my_zz += {steppp}, now = ({my_xx:.3f}, {my_yy:.3f}, {my_zz:.3f})\n")
                elif key == ord('y'):
                    my_zz -= steppp
                    print(f"my_zz -= {steppp}, now = ({my_xx:.3f}, {my_yy:.3f}, {my_zz:.3f})\n")


                gs2.mean3D[:, 0] = gs2.mean3D[:, 0] + now_pos_fld[0] - 1.9987   # x 坐标
                gs2.mean3D[:, 1] = gs2.mean3D[:, 1] + now_pos_fld[1] - 0.3777   # y 坐标
                gs2.mean3D[:, 2] = gs2.mean3D[:, 2] + now_pos_fld[2] - 0.1936   # z 坐标
                # lego_rot_cap = now_rot_cap @ cap2fld_R
                lego_rot_cap = euler_to_rotation_matrix(my_xx, my_yy, my_zz)
                mean3D_bak = gs2.mean3D
                rots_bak = gs2.rots
                gs2.mean3D, gs2.rots = rotate_o(gs2, torch.from_numpy(np.array(lego_rot_cap)).float().to('cuda'))
                # print("now_rot_cap",now_rot_cap)
                # gs2.mean3D[:, 0] -= 1.82101232e+00
                # gs2.mean3D[:, 1] -= 4.55574551e-01
                # gs2.mean3D[:, 2] -= 1.23909457e+00
                
                # R = cap2fld[0, :3, :3].cpu().numpy()
                # R_inv = np.linalg.inv(R)
                # R1 = np.dot(now_rot_cap, np.linalg.inv(init_rot_cap))
                # R2 = np.dot(R, R1)
                # R3 = np.dot(R2, R_inv)

                # gs2.mean3D = torch.matmul(gs2.mean3D, torch.tensor(R3, dtype=torch.float32,device="cuda").T)
                # gs2.mean3D[:, 0] += 1.82101232e+00
                # gs2.mean3D[:, 1] += 4.55574551e-01
                # gs2.mean3D[:, 2] += 1.23909457e+00

                # gs2.rots = rotmat_to_unitquat(new_rot).squeeze(0)
                gaussians_scene.update_scene_from_gaussian(gs2)
                gaussians = gaussians_scene.get_gaussians()
                
                background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")    # 背景是黑色[0,0,0]
                net_image = simple_render(custom_cam, gaussians, background)
                # print("2")
                clamped_image = torch.clamp(net_image, min=0, max=1.0)  # 裁剪操作
                scaled_image = clamped_image * 255                      # 缩放到 [0, 255]
                permuted_image = scaled_image.byte().permute(1, 2, 0)   # permute 维度转换
                contiguous_image = permuted_image.contiguous().cpu()    # 确认 contiguous 和转到 CPU 是否正常
                numpy_image = contiguous_image.numpy()                  # 转为 NumPy 数组
                net_image_bytes = memoryview(numpy_image)               # 创建 memoryview

                network_gui.send(net_image_bytes, ".") 
                # print("1")
                # 先把旋转变回去
                # gs2.mean3D[:, 0] -= 1.82101232e+00
                # gs2.mean3D[:, 1] -= 4.55574551e-01
                # gs2.mean3D[:, 2] -= 1.23909457e+00
                # gs2.mean3D = torch.matmul(gs2.mean3D, torch.tensor(np.linalg.inv(R3), dtype=torch.float32,device="cuda").T)
                # gs2.mean3D[:, 0] += 1.82101232e+00
                # gs2.mean3D[:, 1] += 4.55574551e-01
                # gs2.mean3D[:, 2] += 1.23909457e+00
                # 再把平移变回去
                gs2.mean3D = mean3D_bak
                gs2.rots = rots_bak
                gs2.mean3D[:, 0] -= (now_pos_fld[0] - 1.9987)   # x 坐标
                gs2.mean3D[:, 1] -= (now_pos_fld[1] - 0.3777)   # y 坐标
                gs2.mean3D[:, 2] -= (now_pos_fld[2] - 0.1936)   # z 坐标
                break

curses.wrapper(main)