#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    # 定义激活函数
    def setup_functions(self):
        # 从尺度和旋转参数中去构建3Dgaussian的协方差矩阵
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            # 构建一个同时包含缩放和旋转信息的矩阵
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            # 得到实际的协方差矩阵
            actual_covariance = L @ L.transpose(1, 2)
            # 提取协方差矩阵的下三角和对角线元素
            symm = strip_symmetric(actual_covariance)
            return symm
        
        # 将尺度限制为非负数
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        # 将不透明度限制在0-1的范围内
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    # 一、初始化参数
    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)                      # 中心点位置, 也即3Dgaussian的均值
        self._features_dc = torch.empty(0)              # 第一个球谐系数, 球谐系数用来表示RGB颜色
        self._features_rest = torch.empty(0)            # 其余球谐系数
        self._scaling = torch.empty(0)                  # 尺度
        self._rotation = torch.empty(0)                 # 旋转参数, 四元组
        self._opacity = torch.empty(0)                  # 不透明度
        self.max_radii2D = torch.empty(0)               # 投影到2D时, 每个2D gaussian最大的半径
        self.xyz_gradient_accum = torch.empty(0)        # 3Dgaussian的均值的累积梯度
        self.denom = torch.empty(0)
        self.optimizer = None                           # 上述各参数的优化器
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    # 根据提供的模型参数和训练参数来恢复或设置类实例的状态
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # 三、从点云PCD创建3D gaussian模型
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        # 空间学习率缩放因子
        self.spatial_lr_scale = spatial_lr_scale
        # 将点云中的点坐标转换为PyTorch张量   (P, 3)
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # 将点云中将RGB转换成球谐系数, C0项的系数   (P, 3)，每个颜色通道对应1个球谐系数
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        # 初始化一个用于存储球谐系数的张量  (P, 3, 16), 每个颜色通道有16个球谐系数
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        # 将转换后的球谐系数（C0项）填充到 features 张量的相应位置。
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # distCUDA2 计算点云中的每个点到与其最近的K个点的平均距离的平方
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # 从这些距离计算出每个点的缩放尺度      (P, 3), 每个点在X, Y, Z方向上的尺度
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        # (P, 4), 每个点的旋转参数, 四元组
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # (P, 1), 每个点的不透明度, 初始化为0.1
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))    # (P, 3)
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))   # (P, 1, 3)
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))  # (P, 15, 3)
        self._scaling = nn.Parameter(scales.requires_grad_(True))               # (P, 3)
        self._rotation = nn.Parameter(rots.requires_grad_(True))                # (P, 4)
        self._opacity = nn.Parameter(opacities.requires_grad_(True))            # (P, 1)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")  # (P,)

    # 二、为3D gaussian的各组参数创建optimizer以及lr_scheduler
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense    # 0.01
        # 存储每个3D gaussian的均值xyz的梯度, 用于判断是否对该3D gaussian进行克隆或者切分
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")    # (P, 1)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # (P, 1)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        # 创建optimizer
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # 创建对xyz参数进行学习率调整的scheduler
        # 生成一个连续的学习率衰减函数，这个函数根据训练的进度动态调整学习率。
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    # 五、学习率更新
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # 六、重置不透明度，在训练过程中动态地调整不透明度参数
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
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

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    # 6 动态地调整不透明度参数
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # 删除不符合要求的3D gaussian在self.optimizer中对应的参数(均值、球谐系数、不透明度、尺度、旋转参数)
    # optimizable_tensors保存删除后的优化参数列表
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # 对不符合要求的3D gaussian进行删除
    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    # 拼接新的3D gaussian在self.optimizer中对应的参数(均值、球谐系数、不透明度、尺度、旋转参数)
    # optimizable_tensors保存拼接后的优化参数列表
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # 将挑选出来的3D gaussian的参数拼接到原有的参数之后
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    # 对于那些均值的梯度超过一定阈值且尺度大于一定阈值的3D gaussian进行分割操作
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        # 筛选3D gaussian
        padded_grad = torch.zeros((n_init_points), device="cuda")   # (P,)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)     # (P,)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)      # (2 * P, 3)
        means =torch.zeros((stds.size(0), 3),device="cuda")         # (2 * P, 3)
        samples = torch.normal(mean=means, std=stds)                # (2 * P, 3)
        # 获取选中点的旋转，由四元数转旋转矩阵
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)  # (2 * P, 3, 3)
        # 再以原来3Dgaussian的均值xyz为中心, stds为形状, rots为方向的椭球内随机采样新的3D gaussian
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)  # (2 * P, 3)
        # 由于原来的3D gaussian的尺度过大, 现在将3D gaussian的尺度缩小为原来的1/1.6
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))     # (2 * P, 3)
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)                   # (2 * P, 4)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)           # (2 * P, 1, 3)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)       # (2 * P, 15, 3)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)                     # (2 * P, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        # 将原来的那些均值的梯度超过一定阈值且尺度大于一定阈值的3D gaussian进行删除 (因为已经将它们分割成了两个新的3D gaussian，原先的不再需要了)
        # 前面为条件筛选掩码，后2P个全为False，进入函数取反为True，会保留
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    # 对于那些均值的梯度超过一定阈值且尺度小于一定阈值的3D gaussian进行克隆操作
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        # 筛选3D gaussian
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]                         # (P, 3)
        new_features_dc = self._features_dc[selected_pts_mask]         # (P, 1)
        new_features_rest = self._features_rest[selected_pts_mask]     # (P, 15)
        new_opacities = self._opacity[selected_pts_mask]               # (P, 1)
        new_scaling = self._scaling[selected_pts_mask]                 # (P, 1)
        new_rotation = self._rotation[selected_pts_mask]               # (P, 4)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    # 四、根据梯度对3D gaussian进行增加或删减
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        # 3D gaussian的均值的累积梯度
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # 如果某些3D gaussian的均值的梯度过大且尺度小于一定阈值，说明是欠重建，则对它们进行克隆
        self.densify_and_clone(grads, max_grad, extent)
        # 如果某些3D gaussian的均值的梯度过大且尺度超过一定阈值，说明是过重建，则对它们进行切分
        self.densify_and_split(grads, max_grad, extent)

        # 删除不透明度小于一定阈值的3Dgaussian，用一维数组来存储
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            # 删除2D半径超过2D尺寸阈值的高斯
            big_points_vs = self.max_radii2D > max_screen_size
            # 删除尺度超过一定阈值的高斯
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        # 对不符合要求的3D gaussian进行删除
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    # 根据渲染出来的点统计3D gaussian均值(xyz)的梯度, 用于对3D gaussians的克隆或者切分
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1