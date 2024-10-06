'''
Copyright @ ININ LAMPS

Generalizable 3DGS

Updatable 3DGS class
'''
# standard lib
import os
from dataclasses import dataclass

# framework lib
import numpy as np
import torch
from torch import nn
# 3dparty lib
from plyfile import PlyData, PlyElement

@dataclass
class MiniGaussian:
    '''
    basic 3dgs class
    '''
    active_sh_degree: int
    max_sh_degree: int
    rots: torch.tensor
    opacities: torch.tensor
    mean3D: torch.tensor
    scales: torch.tensor
    features: torch.tensor

class G4DGS:
    '''
    Generalizable 4DGS

    Functions:
        __init__: create G4DGS

        init_scene: init background scene
        update_scene: update dynamic scene
        set_gaussians: directly set all gaussians parameters 

        get_gaussians: read current gaussians
    '''

    def __init__(self, active_sh_degree:int=0, max_sh_degree:int=0):
        # default G3DGS don't need changings about degree
        assert active_sh_degree <= max_sh_degree 

        self.active_sh_degree = active_sh_degree
        self.max_sh_degree = active_sh_degree
        self.reset()

    def reset(self):
        self.rots = None
        self.opacities = None
        self.mean3D = None
        self.scales = None
        self.features = None
        
    def init_scene_from_plyfile(self, ply_file, original_form=False):
        gs = self._load_ply(ply_file, original_form)
        self.bg_rots, self.bg_opacities, self.bg_mean3D, self.bg_scales, self.bg_features = \
                gs.rots, gs.opacities, gs.mean3D, gs.scales, gs.features
                
    def init_scene_from_gaussian(self, ply_file):
        self.bg_rots, self.bg_opacities, self.bg_mean3D, self.bg_scales, self.bg_features = \
                gs.rots, gs.opacities, gs.mean3D, gs.scales, gs.features

    def update_scene_from_plyfile(self, ply_file, original_form=False):
        gs = self._load_ply(ply_file, original_form)
        self.fg_rots, self.fg_opacities, self.fg_mean3D, self.fg_scales, self.fg_features = \
                gs.rots, gs.opacities, gs.mean3D, gs.scales, gs.features

    def update_scene_from_gaussian(self, gs):
        self.fg_rots, self.fg_opacities, self.fg_mean3D, self.fg_scales, self.fg_features = \
                gs.rots, gs.opacities, gs.mean3D, gs.scales, gs.features

    def get_gaussians(self):
        # if self.mean3D is None:
            # generate mini gaussian via background and foreground gaussians
        self.active_sh_degree = self.active_sh_degree
        self.max_sh_degree = self.max_sh_degree
        # print(f"Foreground mean shape: {self.fg_rots.shape}")
        # print(f"Background mean shape: {self.bg_rots.shape}")
        self.rots = torch.cat([self.fg_rots, self.bg_rots], dim=0)                  # 前景和背景的旋转矩阵连接在一起形成总的旋转矩阵。
        self.opacities = torch.cat([self.fg_opacities, self.bg_opacities], dim=0)   # 将前景和背景的透明度连接在一起。
        # print(f"Foreground mean shape: {self.fg_mean3D.shape}")
        # print(f"Background mean shape: {self.bg_mean3D.shape}")
        self.mean3D = torch.cat([self.fg_mean3D, self.bg_mean3D], dim=0)            # 将前景和背景的3D坐标均值连接在一起。
        # print(f"Add mean3d: {self.mean3D[0]}")
        self.scales = torch.cat([self.fg_scales, self.bg_scales], dim=0)            # 将前景和背景的尺度参数连接在一起
        self.features = torch.cat([self.fg_features, self.bg_features], dim=0)      # 将前景和背景的特征参数连接在一起

        mini_gs = MiniGaussian(
            active_sh_degree = self.active_sh_degree,
            max_sh_degree = self.max_sh_degree,
            rots = self.rots,
            opacities = self.opacities,
            mean3D = self.mean3D,
            scales = self.scales,
            features = self.features,)

        # self.reset()

        return mini_gs

    # save and load ply implement
    def construct_list_of_attributes(self, gs):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        f_dc_shape = gs.features[:, 0:1, :].shape
        f_rest_shape = gs.features[:, 1:, :].shape
        for i in range(f_dc_shape[1]*f_dc_shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(f_rest_shape[1]*f_rest_shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(gs.scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(gs.rots.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def _original_3dgs_decoder(self, gs):
        # print("gs.opacities:", gs.opacities)
        # print("gs.scales:", gs.scales)
        mini_gs = MiniGaussian(
            active_sh_degree = self.active_sh_degree,
            max_sh_degree = self.max_sh_degree,
            rots = torch.nn.functional.normalize(gs.rots),
            opacities = torch.sigmoid(gs.opacities),        # 变化1
            mean3D = gs.mean3D,
            scales = torch.exp(gs.scales),                  # 变化2
            features = gs.features)
        # print("mini_gs.opacities:", mini_gs.opacities)
        # print("mini_gs.scales:", mini_gs.scales)
        return mini_gs

    def _save_ply(self, gs:MiniGaussian, path):
        '''
        save 3dgs file in ply form

        Params:
            path: *.ply file path
            gs: MiniGaussian instantce
            origin_form: orininal 3dgs form need extra convert
        '''
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = gs.mean3D.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        opacities = gs.opacities.detach().cpu().numpy()
        scale = gs.scales.detach().cpu().numpy()
        rotation = gs.rots.detach().cpu().numpy()
        f_dc = gs.features.transpose(1, 2)[:, :, 0:1].detach().flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = gs.features.transpose(1, 2)[:, :, 1:].flatten(start_dim=1).contiguous().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(gs)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def _load_ply(self, path:str, original_form:bool=False):
        '''
        load 3dgs file in ply form

        Params:
            path: *.ply file path
            origin_form: orininal 3dgs form need extra convert
        '''
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # 提取旋转矩阵
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # 以下是原版高斯，用于对比，实际没什么用
        # rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        # rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        # rots = np.zeros((xyz.shape[0], len(rot_names)))
        # for idx, attr_name in enumerate(rot_names):
        #     rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        # self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        # self.active_sh_degree = self.max_sh_degree


        # 以下是Mini_gaussian
        features_dc = torch.tensor(features_dc, dtype=torch.float, device="cuda")
        features_extra = torch.tensor(features_extra, dtype=torch.float, device="cuda")
        features = torch.cat([features_dc, features_extra], dim=-1).transpose(1, 2)
        mini_gs = MiniGaussian(
            active_sh_degree = self.active_sh_degree,
            max_sh_degree = self.max_sh_degree,
            rots = torch.tensor(rots, dtype=torch.float, device="cuda"),
            opacities = torch.tensor(opacities, dtype=torch.float, device="cuda"),
            mean3D = torch.tensor(xyz, dtype=torch.float, device="cuda"),
            scales = torch.tensor(scales, dtype=torch.float, device="cuda"),
            features = features.contiguous(),)
        if original_form:
            mini_gs = self._original_3dgs_decoder(mini_gs)

        return mini_gs
    
    def save_npz(self, path):
        # 需要以dict的形式保存；
        '''
        需要保存的参数：
        _xyz, _features_dc, _features_rest, _opacity, _scaling, _rotation, sem_labels
        '''
        # dict_to_save = {
        #     "xyz": self._xyz.detach().cpu().numpy(),
        #     "f_dc": self._features_dc.detach().cpu().numpy(),
        #     "f_rest": self._features_rest.detach().cpu().numpy(),
        #     "opacity": self._opacity.detach().cpu().numpy(),
        #     "scaling": self._scaling.detach().cpu().numpy(),
        #     "rotation": self._rotation.detach().cpu().numpy()
        # }
        dict_to_save = {
            "xyz": self.bg_mean3D.detach().cpu().numpy(),
            "opacity": self.bg_opacities.detach().cpu().numpy(),
            "scales": self.bg_scales.detach().cpu().numpy(),
            "rotation": self.bg_rots.detach().cpu().numpy(),
            "shs": self.bg_features.detach().cpu().numpy()
        }

        # if self.sem_labels is not None:
        #     dict_to_save["sem_labels"] = self.sem_labels.detach().cpu().numpy()

        # 保存所有的mapping_cams中的R和T作为一个列表
        # R_list = [cam.R.numpy() for cam in mapping_cams]
        # T_list = [cam.t.numpy() for cam in mapping_cams]
        # dict_to_save["R_list"] = np.array(R_list)
        # dict_to_save["T_list"] = np.array(T_list)

        np.savez(path, **dict_to_save)

    def save_bg_scene(self, path):
        mini_gs = MiniGaussian(
            active_sh_degree = self.active_sh_degree,
            max_sh_degree = self.max_sh_degree,
            rots = self.bg_rots,
            opacities = self.bg_opacities,
            mean3D = self.bg_mean3D,
            scales = self.bg_scales,
            features = self.bg_features,)

        self._save_ply(mini_gs, path)
        print(f"File {path} saved")

    def save_fg_scene(self, path):
        mini_gs = MiniGaussian(
            active_sh_degree = self.active_sh_degree,
            max_sh_degree = self.max_sh_degree,
            rots = self.fg_rots,
            opacities = self.fg_opacities,
            mean3D = self.fg_mean3D,
            scales = self.fg_scales,
            features = self.fg_features,)

        self._save_ply(mini_gs, path)
        print(f"File {path} saved")

    def save_full_scene(self, path):
        mini_gs = MiniGaussian(
            active_sh_degree = self.active_sh_degree,
            max_sh_degree = self.max_sh_degree,
            rots = torch.cat([self.fg_rots, self.bg_rots], dim=0),
            opacities = torch.cat([self.fg_opacities, self.bg_opacities], dim=0),
            mean3D = torch.cat([self.fg_mean3D, self.bg_mean3D], dim=0),
            scales = torch.cat([self.fg_scales, self.bg_scales], dim=0),
            features = torch.cat([self.fg_features, self.bg_features], dim=0))

        self._save_ply(mini_gs, path)
        print(f"File {path} saved")
