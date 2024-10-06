import torch
import traceback
import socket
import json
#  from scene.cameras import MiniCam

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

# for debug
foo_bar_cam = MiniCam(
        #  width=800, height=800,
        width=400, height=400,
        fovy=0.6911112070083618, fovx=0.6911112070083618, 
        znear=0.008999999612569809, zfar=1100.0, 
        world_view_transform=torch.tensor(
        [[-1.0000e+00,  0.0000e+00, -0.0000e+00, -0.0000e+00],
        [-0.0000e+00,  7.3411e-01, -6.7903e-01,  0.0000e+00],
        [ 0.0000e+00, -6.7903e-01, -7.3411e-01, -0.0000e+00],
        [-0.0000e+00, -2.3842e-07,  4.0311e+00,  1.0000e+00]]).cuda(), 

        #  world_view_transform=torch.eye(4).cuda(),
        full_proj_transform=torch.tensor(
        [[-2.7778e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00],
        [ 0.0000e+00,  2.0392e+00, -6.7904e-01, -6.7903e-01],
        [ 0.0000e+00, -1.8862e+00, -7.3412e-01, -7.3411e-01],
        [ 0.0000e+00, -6.6227e-07,  4.0132e+00,  4.0311e+00]]).cuda())

host = "127.0.0.1"
port = 6009

conn = None
addr = None

listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def init(wish_host, wish_port):
    global host, port, listener
    host = wish_host
    port = wish_port
    listener.bind((host, port))
    listener.listen()
    listener.settimeout(0)

def try_connect():
    global conn, addr, listener
    try:
        conn, addr = listener.accept()
        print(f"\nConnected by {addr}")
        conn.settimeout(None)
    except Exception as inst:
        pass
            
def read():
    global conn
    messageLength = conn.recv(4)
    messageLength = int.from_bytes(messageLength, 'little')
    message = conn.recv(messageLength)
    return json.loads(message.decode("utf-8"))

def send(message_bytes, verify):
    global conn
    if message_bytes != None:
        conn.sendall(message_bytes)
    conn.sendall(len(verify).to_bytes(4, 'little'))
    conn.sendall(bytes(verify, 'ascii'))

def receive():
    message = read()
    # print("Received message:", message)
    width = message["resolution_x"]
    height = message["resolution_y"]
    if width == 0 or height == 0:
        print("Invalid resolution: width or height is zero.")

    if width != 0 and height != 0:
        try:
            do_training = bool(message["train"])
            fovy = message["fov_y"]
            fovx = message["fov_x"]
            znear = message["z_near"]
            zfar = message["z_far"]
            do_shs_python = bool(message["shs_python"])
            do_rot_scale_python = bool(message["rot_scale_python"])
            keep_alive = bool(message["keep_alive"])
            scaling_modifier = message["scaling_modifier"]
            world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
            world_view_transform[:,1] = -world_view_transform[:,1]
            world_view_transform[:,2] = -world_view_transform[:,2]
            full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
            full_proj_transform[:,1] = -full_proj_transform[:,1]
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
        except Exception as e:
            print("Error processing message:", message)
            traceback.print_exc()
            raise e
        return custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifier
    else:
        return None, None, None, None, None, None
