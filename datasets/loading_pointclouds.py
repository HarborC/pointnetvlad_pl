#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np

def get_queries_dict(filename):
    with open(filename, 'rb') as handle:
        queries = pickle.load(handle)
        print("Queries Loaded.")
        return queries


def load_pc_file(filename, point_num = 4096):
    # returns Nx3 matrix
    pc = np.load(filename)
    # print(pc)

    # print(pc)
    if(pc.shape[0] != point_num):
        print("Error in pointcloud shape")
        return np.array([])

    # pc = np.reshape(pc,(pc.shape[0]//3, 3))
    return pc


def load_pc_files(filenames, point_num = 4096):
    pcs = []
    for filename in filenames:
        pc = load_pc_file(filename)
        if(pc.shape[0] != point_num):
            continue
        pcs.append(pc)
    pcs = np.array(pcs)
    return pcs


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        #-90 to 90
        rotation_angle = (np.random.uniform()*np.pi) - np.pi/2.0
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(
            shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.005, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

if __name__ == "__main__":
    # database_path_prex = "/media/s1/cjg/dataset/GRP/TRAIN"
    # generate_tuples(database_path_prex)
    path = "/media/s1/cjg/dataset/GRP/TRAIN/pretrained/0.npy"
    print(load_pc_file(path))