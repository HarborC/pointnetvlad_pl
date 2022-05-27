import torch
import torch.nn as nn
from .netvlad import NetVLADLoupe
from .pointnet import PointNetfeat
from .point_transformer import PointTransformerBackbone

class PointNetVlad(nn.Module):
    def __init__(self, num_points=4096, output_dim=256, emb_dims=1024):
        super(PointNetVlad, self).__init__()
        self.backbone = PointNetfeat(num_points=num_points, global_feat=False,
                                     feature_transform=False, max_pool=False, emb_dims=emb_dims)
        self.netvlad = NetVLADLoupe(feature_size=emb_dims, max_samples=num_points, cluster_size=64,
                                    output_dim=output_dim, gating=True, add_batch_norm=True)

    def forward(self, x):
        # print("input x: ",x.shape)
        x = self.backbone(x)
        # print("point_net x: ", x.shape)
        x = self.netvlad(x)
        # print("netvlad x: ", x.shape) [B, output_dim]
        return x