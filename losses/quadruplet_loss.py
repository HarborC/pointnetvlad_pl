import torch
from torch import nn
from loss_util import best_pos_distance

class QuadrupletLoss(nn.Module):
    def __init__(self, margin_1, margin_2, use_min=False, lazy=False, ignore_zero_loss=False):
        super(QuadrupletLoss, self).__init__()
        self.margin_1 = margin_1
        self.margin_2 = margin_2
        self.use_min = use_min
        self.lazy = lazy
        self.ignore_zero_loss = ignore_zero_loss

    def forward(self, q_vec, pos_vecs, neg_vecs, other_neg):
        min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)

        # PointNetVLAD official code use min_pos, but i think max_pos should be used
        if self.use_min:#获得正样本最小的距离还是最大
            positive = min_pos
        else:
            positive = max_pos

        num_neg = neg_vecs.shape[1]
        batch = q_vec.shape[0]
        query_copies = q_vec.repeat(1, int(num_neg), 1)
        positive = positive.view(-1, 1)
        positive = positive.repeat(1, int(num_neg))

        loss = self.margin_1 + positive - ((neg_vecs - query_copies) ** 2).sum(2)
        loss = loss.clamp(min=0.0)
        # 是否只看max
        if self.lazy:
            triplet_loss = loss.max(1)[0]
        else:
            triplet_loss = loss.sum(1)
        # 是否忽略为0的loss
        if self.ignore_zero_loss:
            # gt 若大于1e-16则为1
            hard_triplets = torch.gt(triplet_loss, 1e-16).float()
            num_hard_triplets = torch.sum(hard_triplets)
            triplet_loss = triplet_loss.sum() / (num_hard_triplets + 1e-16)
        else:
            triplet_loss = triplet_loss.mean()

        other_neg_copies = other_neg.repeat(1, int(num_neg), 1)
        second_loss = self.margin_2 + positive - ((neg_vecs - other_neg_copies) ** 2).sum(2)
        second_loss = second_loss.clamp(min=0.0)
        if self.lazy:
            second_loss = second_loss.max(1)[0]
        else:
            second_loss = second_loss.sum(1)
        # 是否忽略为0的loss
        if self.ignore_zero_loss:
            # gt 若大于1e-16则为1
            hard_second = torch.gt(second_loss, 1e-16).float()
            num_hard_second = torch.sum(hard_second)
            second_loss = second_loss.sum() / (num_hard_second + 1e-16)
        else:
            second_loss = second_loss.mean()

        total_loss = triplet_loss + second_loss
        return total_loss
