import sys
import os
import pickle
import random
from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

def construct_query_dict(traj, filename):
    tree = KDTree(traj['xy'])
    ind_nn = tree.query_radius(traj['xy'], r=12.5)
    ind_r = tree.query_radius(traj['xy'], r=50)
    queries = {}
    print(len(ind_nn))
    for i in tqdm(range(len(ind_nn)), desc = 'construct query: '):
        query = (traj['file'])[i]
        positives = np.setdiff1d(ind_nn[i], [i]).tolist()
        list_neg = np.setdiff1d(range(len(ind_nn)), ind_r[i]).tolist()
        # if len(list_neg) > 4001:
        #     random.shuffle(list_neg)
        #     list_neg = list_neg[:4001]
        negatives = list_neg
        random.shuffle(negatives)
        queries[i] = {"query":query,
                      "positives":positives,"negatives":negatives}

    with open(filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)


import sys 
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from datasets.UgvLoader import UgvLoader
from datasets.loading_pointclouds import get_queries_dict

def generate_tuples(database_path_prex):
    if not os.path.exists(os.path.join(database_path_prex, "queries.pickle")):
        dirs_temp = os.listdir(database_path_prex)
        dirs = []
        for dir in dirs_temp:
            if Path(os.path.join(database_path_prex, dir)).is_dir() and 'pretrained' not in dir:
                dirs.append(dir)

        traj = {'xy': [], 'file': []}
        for dir in tqdm(dirs, desc = 'load dataset: '):
            database_path = os.path.join(database_path_prex, dir)
            ugv_loader = UgvLoader(database_path)

            for idx in range(len(ugv_loader)):
                traj['xy'].append(ugv_loader.get_translation(idx))
                traj['file'].append(ugv_loader.get_pcd_path(idx))
            # print(database_path, ' is finished. ')
            
        traj['xy'] = np.vstack(traj['xy'])
        traj['xy'] = (traj['xy'])[:,:2]

        construct_query_dict(traj, os.path.join(database_path_prex, "queries.pickle"))

import open3d as o3d

def ground_filter(pcl_points):
    # print("do ground filter")
    points = np.array(pcl_points.points)
    center = np.mean(points, axis=0)

    down_pcl_points = o3d.geometry.PointCloud()
    down_pcl_points.points = o3d.utility.Vector3dVector(points[points[:,2]<center[2]])
    pcl_points.points = o3d.utility.Vector3dVector(points[points[:,2]>=center[2]])
    plane_model, inliers = down_pcl_points.segment_plane(distance_threshold=0.01,
                                                         ransac_n=5,
                                                         num_iterations=10000)
    # print(plane_model)
    down_pcl_points1 = down_pcl_points.select_by_index(inliers, invert=False)
    down_pcl_points2 = down_pcl_points.select_by_index(inliers, invert=True)
    down_pcl_points_ = np.array(down_pcl_points1.points)
    down_pcl_points2_ = np.array(down_pcl_points2.points)
    max_z = np.max(down_pcl_points2_[:,2])
    down_pcl_points1.points = o3d.utility.Vector3dVector(down_pcl_points_[down_pcl_points_[:,2]>=max_z])

    pcd_data = pcl_points + down_pcl_points1

    return pcd_data


def get_fix_points(points, points_num=4096, voxel_resolution=0.1):
    if len(points.points) <= points_num:
        return points

    downpcd, _, indexes = points.voxel_down_sample_and_trace(voxel_resolution, \
                                             min_bound=points.get_min_bound()-0.5, \
                                             max_bound=points.get_max_bound()+0.5)

    if len(downpcd.points) > points_num:
        points_t = np.array(downpcd.points)
        n = np.random.choice(len(downpcd.points), points_num, replace=False)
        downpcd.points = o3d.utility.Vector3dVector(points_t[n])
    elif len(downpcd.points) < points_num:
        dpc_indexes = []
        for vx in indexes:
            dpc_indexes.append(int(vx[0]))
        print(dpc_indexes)
        other_indexes = []
        for m in range(len(points.points)):
            if m not in dpc_indexes:
                other_indexes.append(m)
        print(other_indexes)
        n = random.sample(other_indexes, points_num - len(downpcd.points))
        add_points = o3d.geometry.PointCloud()
        add_points.points = o3d.utility.Vector3dVector(points.points[n])
        downpcd = downpcd + add_points
    
    return downpcd


def process_point_cloud(pcd_data, points_num=4096, voxel_resolution=0.1, filter_ground=True):
    if filter_ground:
        pcd_data = ground_filter(pcd_data)
    
    ds_pcd = get_fix_points(pcd_data, points_num, voxel_resolution)

    # * get raw point cloud
    pcd = np.asarray(ds_pcd.points, dtype=np.float32)

    # normalize points
    centroid = np.mean(pcd, axis = 0)
    pcd = pcd - centroid
    # m = np.max(np.sqrt(np.sum(downpoints ** 2, axis = 1)))
    # downpoints = downpoints / m

    return pcd


def generate_queries(database_path_prex, points_num=4096, voxel_resolution=0.1, filter_ground=True):
    pretrained_database_path = os.path.join(database_path_prex, "pretrained")

    if not os.path.exists(os.path.join(database_path_prex, "queries.pickle")):
        print("no queries_dict")
        exit(0)
    
    queries_dict = get_queries_dict(os.path.join(database_path_prex, "queries.pickle"))
    if not Path(pretrained_database_path).exists():
        os.makedirs(pretrained_database_path)

        for i in tqdm(range(len(queries_dict.keys())), desc = 'generate pretrained lidar data: '):
            query_item = queries_dict[i]
            pcd_data = o3d.io.read_point_cloud(query_item["query"])

            processed_pcl = process_point_cloud(pcd_data, points_num, voxel_resolution, filter_ground)

            # import matplotlib.pyplot as plt
            # fig = plt.figure(figsize=(10, 5))
            # ax_pc = fig.add_subplot(111, projection='3d')
            # ax_pc.scatter(
            #     processed_pcl[:, 0], processed_pcl[:, 1], processed_pcl[:, 2], c=processed_pcl[:, 2], s=3, linewidths=0,
            # )
            # ax_pc.set_xlabel('x')
            # ax_pc.set_ylabel('y')
            # ax_pc.set_xlim(-50.0, 50.0)
            # ax_pc.set_ylim(-50.0, 50.0)
            # plt.show()

            new_path = os.path.join(pretrained_database_path, str(i) + '.npy')
            np.save(new_path, processed_pcl)
            queries_dict[i]["query"] = new_path

        os.remove(os.path.join(database_path_prex, "queries.pickle"))
        with open(os.path.join(database_path_prex, "queries.pickle"), 'wb') as handle:
            pickle.dump(queries_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return queries_dict


def pointcloud_process_test(path):
    points_num = 4096
    voxel_resolution = 0.1
    pcd_data = o3d.io.read_point_cloud(path)
    import matplotlib.pyplot as plt
    show_pcd = np.asarray(pcd_data.points)
    fig = plt.figure(figsize=(10, 5))
    ax_pc = fig.add_subplot(141, projection='3d')
    ax_pc.scatter(
        show_pcd[:, 0], show_pcd[:, 1], show_pcd[:, 2], c=show_pcd[:, 2], s=3, linewidths=0,
    )
    ax_pc.set_xlabel('x')
    ax_pc.set_ylabel('y')
    ax_pc.set_xlim(-50.0, 50.0)
    ax_pc.set_ylim(-50.0, 50.0)

    processed_pcl = process_point_cloud(pcd_data, points_num, voxel_resolution, True)
    ax_pc1 = fig.add_subplot(142, projection='3d')
    ax_pc1.scatter(
        processed_pcl[:, 0], processed_pcl[:, 1], processed_pcl[:, 2], c=processed_pcl[:, 2], s=3, linewidths=0,
    )
    ax_pc1.set_xlabel('x')
    ax_pc1.set_ylabel('y')
    ax_pc1.set_xlim(-50.0, 50.0)
    ax_pc1.set_ylim(-50.0, 50.0)

    processed_pcl_test = process_point_cloud(pcd_data, points_num, voxel_resolution, False)
    ax_pc2 = fig.add_subplot(143, projection='3d')
    ax_pc2.scatter(
        processed_pcl_test[:, 0], processed_pcl_test[:, 1], processed_pcl_test[:, 2], c=processed_pcl_test[:, 2], s=3, linewidths=0,
    )
    ax_pc2.set_xlabel('x')
    ax_pc2.set_ylabel('y')
    ax_pc2.set_xlim(-50.0, 50.0)
    ax_pc2.set_ylim(-50.0, 50.0)
    plt.show()


if __name__ == "__main__":
    # database_path_prex = "/media/s1/cjg/dataset/GRP/TRAIN"
    # generate_tuples(database_path_prex)
    path = "/media/s1/cjg/dataset/GRP/TRAIN/train_1/000001.pcd"
    pointcloud_process_test(path)