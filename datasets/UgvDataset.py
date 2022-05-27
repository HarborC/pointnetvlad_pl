from torch.utils.data import Dataset
import os
import numpy as np
import torch
import random
from time import time

from query_generation.generate_tuples import generate_tuples, generate_queries
from loading_pointclouds import load_pc_file, load_pc_files

class UgvDataset(Dataset):
    def __init__(self, args, type='train', val_ratio=0.8):
        self.root_dir=args.root_dir
        self.positives_per_query=args.positives_per_query
        self.negatives_per_query=args.negatives_per_query
        self.num_points = args.num_points 

        generate_tuples(self.root_dir)
        self.queries = generate_queries(self.root_dir, self.num_points, \
                                        args.voxel_resolution, args.filter_ground)

        data_len_ = len(self.queries.keys())
        use_idxes_ = list(range(data_len_))
        random.seed(0)
        random.shuffle(use_idxes_)
        if type == 'train':
            self.batch_size = args.batch_size
            self.data_len = int(data_len_ * val_ratio)
            self.data_len = int(self.data_len / self.batch_size) * self.batch_size
            self.use_idxes = use_idxes_[:self.data_len]
        elif type == 'val':
            self.batch_size = args.batch_size 
            self.data_len = data_len_ - int(data_len_ * val_ratio) 
            self.data_len = int(self.data_len / self.batch_size) * self.batch_size
            self.use_idxes = use_idxes_[-self.data_len:]    
    
        self.last = []
        self.sample = []

        print('Load UgvDataset')


    def get_default(self):
        if self.last == []:
            print("wrong")
            return False
        else:
            self.sample = [self.last[0], self.last[1], self.last[2], self.last[3]]
            return True


    def __getitem__(self, idx):
        item = self.use_idxes[idx]
        if (len(self.queries[item]["positives"]) < self.positives_per_query):
            if self.get_default():
                return self.sample
        
        q_tuples = self.get_query_tuple(self.queries[item], self.positives_per_query, self.negatives_per_query, self.queries)
            
        query = np.expand_dims(np.array(q_tuples[0], dtype=np.float32), axis=0)
        other_neg = np.expand_dims(np.array(q_tuples[3], dtype=np.float32), axis=0)
        positives = np.array(q_tuples[1], dtype=np.float32)
        negatives = np.array(q_tuples[2], dtype=np.float32)

        if (len(query.shape) != 3):
            if self.get_default():
                return self.sample
                
        self.last = [query, positives, negatives, other_neg]
        return query.astype('float32'), positives.astype('float32'), negatives.astype('float32'), other_neg.astype('float32')


    def __len__(self):
        return len(self.use_idxes)

    
    def get_query_tuple(self, dict_value, num_positive, num_negative, all_dict_value, hard_neg=[], other_neg=True):
        
        show_use_time = False
        if show_use_time:
            start = time()
        
        query = load_pc_file(dict_value["query"], self.num_points)
        
        pos_files = []
        random.shuffle(dict_value["positives"])
        for i in range(num_positive):
            pos_idx = dict_value["positives"][i]
            pos_files.append(all_dict_value[pos_idx]["query"])
        positives = load_pc_files(pos_files, self.num_points)

        neg_files = []
        neg_indices = []
        random.shuffle(dict_value["negatives"])
        for neg_idx in hard_neg:
            neg_files.append(all_dict_value[neg_idx]["query"])
            neg_indices.append(neg_idx)

        for i in range(len(dict_value["negatives"])):
            if len(neg_files) >= num_negative:
                break
            neg_idx = dict_value["negatives"][i]
            if not neg_idx in hard_neg:
                neg_files.append(all_dict_value[neg_idx]["query"])
                neg_indices.append(neg_idx)
        negatives = load_pc_files(neg_files, self.num_points)

        if show_use_time:
            print("load time: ", time()-start)

        if other_neg is False:
            return [query, positives, negatives]
        else:
            neighbors = []
            for pos in dict_value["positives"]:
                neighbors.append(pos)
            for neg in neg_indices:
                for pos in all_dict_value[neg]["positives"]:
                    neighbors.append(pos)

            possible_negs = list(set(all_dict_value.keys())-set(neighbors))
            random.shuffle(possible_negs)
            if(len(possible_negs) == 0):
                return [query, positives, negatives, np.array([])]
            neg2 = load_pc_file(all_dict_value[possible_negs[0]]["query"], self.num_points)

            return [query, positives, negatives, neg2]
