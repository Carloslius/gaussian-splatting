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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        """ 
            寻找是否有训练过的记录
            如果没有则为初次训练, 需要从COLMAP创建的点云中初始化每个点对应的3D gaussian
            以及将每张图片对应的相机参数dump到`cameras.json`文件中
            如果有则直接赋值
        """
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        # 从COLMAP或Blender中读取每张图片, 以及每张图片对应的相机内外参
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            # class SceneInfo(NamedTuple):
            #       point_cloud: BasicPointCloud    点云数据，包括点的位置、颜色和法线信息
            #       train_cameras: list             每个train_camera包含了一张Image的信息
            #       test_cameras: list              默认为空
            #       nerf_normalization: dict        {"translate": translate, "radius": radius}
            #       ply_path: str                   初始点云文件路径
            # 其中：BasicPointCloud(points=positions, colors=colors, normals=normals)
            # 可把scene_info中的train_cameras当作camera_infos
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        # 将每张图片对应的相机参数存储到cameras.json文件中
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # 随机打乱所有图片和对应相机的顺序
        # class CameraInfo(NamedTuple):
        #     uid: int          image_id
        #     R: np.array
        #     T: np.array
        #     FovY: np.array
        #     FovX: np.array
        #     image: np.array
        #     image_path: str
        #     image_name: str
        #     width: int
        #     height: int
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        # 所有相机的平均中心点位置，以及平均中心点到最远camera的距离
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 对每个分辨率比例，加载并存储训练和测试用的相机列表
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        # 如果是初次训练, 则从COLMAP创建的点云中初始化每个点对应的3D gaussian
        # 否则直接从之前保存的模型文件中读取3D gaussian
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    # train_cameras = camera_list, 存储着Camera，一张图片对应一个Camera
    # return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
    #           FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
    #           image=gt_image, gt_alpha_mask=loaded_mask,
    #           image_name=cam_info.image_name, uid=id, data_device=args.data_device)
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]