import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class STN3d(nn.Module):
    def __init__(self, num_points=2500, k=3, use_bn=True):
        super(STN3d, self).__init__()
        self.k = k
        self.kernel_size = 3 if k == 3 else 1
        self.channels = 1 if k == 3 else k
        self.num_points = num_points
        self.use_bn = use_bn
        self.conv1 = torch.nn.Conv2d(self.channels, 64, (1, self.kernel_size))
        self.conv2 = torch.nn.Conv2d(64, 128, (1,1))
        self.conv3 = torch.nn.Conv2d(128, 1024, (1,1))
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.zero_()
        self.relu = nn.ReLU()

        if use_bn:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        if self.use_bn:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        if self.use_bn:
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).astype(np.float32))).view(
            1, self.k*self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_points=2500, global_feat=True, feature_transform=False, max_pool=True, emb_dims=1024):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points=num_points, k=3, use_bn=False)
        self.feature_trans = STN3d(num_points=num_points, k=64, use_bn=False)
        self.apply_feature_trans = feature_transform
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, (1, 1))
        self.conv3 = torch.nn.Conv2d(64, 64, (1, 1))
        self.conv4 = torch.nn.Conv2d(64, 128, (1, 1))
        self.conv5 = torch.nn.Conv2d(128, emb_dims, (1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(emb_dims)
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.num_points = num_points
        self.global_feat = global_feat
        self.max_pool = max_pool
        self.emb_dims = emb_dims

    # [B,1,num,3]
    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        # print("trans: ", trans.shape) # [B,3,3]
        # 去掉“1”维度然后相乘
        x = torch.matmul(torch.squeeze(x,dim=1), trans)
        # print("x: ", x.shape) [B,num,3]
        x = x.view(batchsize, 1, -1, 3)
        # print("x: ", x.shape) [B,1,num,3]
        x = F.relu(self.bn1(self.conv1(x)))
        # print("x: ", x.shape) [B,64,num,1]
        x = F.relu(self.bn2(self.conv2(x)))
        # print("x: ", x.shape) [B,64,num,1]
        pointfeat = x
        if self.apply_feature_trans:
            f_trans = self.feature_trans(x)
            x = torch.squeeze(x)
            if batchsize == 1:
                x = torch.unsqueeze(x, 0)
            x = torch.matmul(x.transpose(1, 2), f_trans)
            x = x.transpose(1, 2).contiguous()
            x = x.view(batchsize, 64, -1, 1)
        x = F.relu(self.bn3(self.conv3(x)))
        # print("x: ", x.shape) [B,64,num,1]
        x = F.relu(self.bn4(self.conv4(x)))
        # print("x: ", x.shape) [B,128,num,1]
        x = self.bn5(self.conv5(x))
        # print("x: ", x.shape) [B,emb_dims,num,1]
        if not self.max_pool:
            return x
        else:
            x = self.mp1(x)
            x = x.view(-1, self.emb_dims)
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, self.emb_dims, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
