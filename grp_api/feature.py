import sys 
import os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
sys.path.append(os.path.dirname(__file__) + os.sep + './')

from models.pointnetvlad import PointNetVlad
from query_generation.generate_tuples import process_point_cloud
import open3d as o3d
import numpy as np
import torch
from utils import load_ckpt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PointNetVladFeature:
    def __init__(
        self,
        pretrained_path,
        prefixes_to_ignore,
        num_points = 4096,
        output_dim = 256,
        emb_dims = 1024
    ):
        self.model = PointNetVlad(num_points, output_dim, emb_dims)
        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("use cuda!")
        else:
            self.model = self.model.cpu()
            print("use cpu!")

        load_ckpt(self.model, pretrained_path, prefixes_to_ignore)

    def preprocess(self, query):
        new_query = []

        for q in query:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(q)
            new_q = process_point_cloud(pcd, points_num=4096, voxel_resolution=0.1, filter_ground=False)
            new_query.append(new_q)

        new_query = np.array(new_query)

        return new_query

    def infer_data(self, query):
        new_query = self.preprocess(query)
        with torch.no_grad():
            feed_tensor = torch.from_numpy(new_query).float()
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)
            out = self.model(feed_tensor)

        query_desc = out.detach().cpu().numpy()
        return query_desc.reshape(-1)