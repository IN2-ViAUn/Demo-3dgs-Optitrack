'''
Test save and load ply file
'''
import setproctitle
proc_title = "JTJ_mini_3DGS_test"
setproctitle.setproctitle(proc_title)

import sys
import os
import time 
from argparse import ArgumentParser

import torch

import network_gui
from G4DGS.G4DGS import G4DGS
from render import simple_render


if __name__ == '__main__':

    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--workspace', type=str, default="./output")
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--path', type=str, default="../../../datasets/self_blender_rbgd/rgbds720P/")
    args = parser.parse_args(sys.argv[1:])

    # arguments
    path = args.path
    ip = args.ip
    port = args.port
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create 3dgs
    gaussians_scene =  G4DGS(3,3)

    gaussians_scene.init_scene_from_plyfile(os.path.join(args.workspace, 'bg.ply'))

    #  bg_color =  [1, 1, 1]
    bg_color =  [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # init remote GUI
    network_gui.init(ip, port)

    frame_no = 0
    while True: # mini dataset only have 20 frames
        # infinite loop
        frame_no = frame_no+1 if frame_no < 20 -1 else 0
        gaussians_scene.update_scene_from_plyfile(os.path.join(args.workspace, f'fg{frame_no}.ply'))
        #  gaussians_scene.save_fg_scene(os.path.join(args.workspace, f'fg{frame_no}.ply'))
        #  print(f"Forground saved at {os.path.join(args.workspace, f'fg{frame_no}.ply')}")

    # connecting SIBR Veiwer
        if network_gui.conn == None:
        # connecting
            print('Try to connect...')
            network_gui.try_connect()
        while network_gui.conn != None:
            print('Connected')
            #  try:
                #  custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
            net_image_bytes = None
            custom_cam, _, _, _, keep_alive, scaling_modifer = network_gui.receive()
            if custom_cam != None:
                # rendering
                net_image = simple_render(custom_cam, gaussians_scene.get_gaussians(), background)
                net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, ".")
                break
            #  except Exception as e:
                #  network_gui.conn = None

