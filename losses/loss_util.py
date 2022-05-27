import numpy as np
import math

def best_pos_distance(query, pos_vecs):
    num_pos = pos_vecs.shape[1]
    query_copies = query.repeat(1, int(num_pos), 1)
    diff = ((pos_vecs - query_copies) ** 2).sum(2)
    min_pos, _ = diff.min(1)
    max_pos, _ = diff.max(1)
    return min_pos, max_pos