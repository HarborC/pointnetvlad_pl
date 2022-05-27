'''
Filename: /home/maxtom/codespace/GPR_Competition/src/gpr/dataloader/base_loader.py
Path: /home/maxtom/codespace/GPR_Competition/src/gpr/dataloader
Created Date: Sunday, March 6th 2022, 9:26:53 pm
Author: maxtom

Copyright (c) 2022 Your Company
'''
import os
from typing import Tuple
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from gpr.dataloader.BaseLoader import BaseLoader

# NOTE: This dataset loader is not complete now !!


class UgvLoader(BaseLoader):
    def __init__(
        self,
        dir_path: str,
        resolution: float = 0.5,
    ):
        """Data loader for the UGV Dataset.
        Args:
            image_size [int, int]: set image resolution
            resolution [float]: resolution for point cloud voxels
        """
        super().__init__(dir_path)
        self.resolution = resolution

        self.file_num = 0
        files = os.listdir(dir_path)
        for file in files:
            if file[0] != '.' and os.path.splitext(file)[-1] == '.pcd':
                self.file_num = self.file_num + 1

        self.file_num = self.file_num

        self.poses = []
        for i in range(self.file_num):
            file_pose_path = str(i+1).zfill(6) + "_pose6d.npy"
            pose = np.load(self.dir_path + '/' + file_pose_path)
            self.poses.append(pose[0:6])

    def __len__(self) -> int:
        """Return the number of frames in this dataset"""
        return self.file_num

    def __getitem__(self, idx: int):
        '''Return the query data (Image, LiDAR, etc)'''
        pcd, sph, top = self.get_point_cloud(idx)
        return {'pcd': pcd, 'sph': sph, 'top': top}

    def get_pose(self, frame_id: int) -> np.ndarray:
        """Get the pose (4*4 transformation matrix) at the `frame_id` frame.
        Args:
            frame_id: the index of current frame
        Returns:
            pose: numpy.ndarray([[R, t], [0, 1]]), of size (4, 4)
        Raise:
            ValueError: If this dataset doesn't have poses
        """
        if self.poses is None:
            raise ValueError(
                'This dataset does NOT have poses. '
                'Maybe it is used for testing/query'
            )

        pose6d = self.poses[frame_id]  # size (6,)
        rot_matrix = R.from_euler('xyz', pose6d[3:]).as_matrix()
        trans_vector = pose6d[:3].reshape((3, 1))

        trans_matrix = np.identity(4)
        trans_matrix[:3, :3] = rot_matrix
        trans_matrix[:3, 3:] = trans_vector

        return trans_matrix

    def get_pcd_path(self, frame_id: int):
        return os.path.join(self.dir_path, str(frame_id+1).zfill(6) + '.pcd')

    def get_point_cloud(self, frame_id: int):
        '''Get the point cloud at the `frame_id` frame.
        Raise ValueError if there is no point cloud in the dataset.
        return -> o3d.geometry.PointCloud
        '''
        pcl_path = self.get_pcd_path(frame_id)
        pcd_data = o3d.io.read_point_cloud(pcl_path)
        ds_pcd = pcd_data.voxel_down_sample(self.resolution)

        # * get raw point cloud
        pcd = np.asarray(ds_pcd.points)

        return pcd
