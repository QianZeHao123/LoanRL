import numpy as np
import torch
from collections import Counter
from scipy.stats import entropy


class HashTable():
    def __init__(self, hash_size, inp_dimensions, action_num=5):
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.action_num = action_num
        self.hash_table = dict()
        for i in range(2 ** hash_size):
            self.hash_table[i] = np.zeros(self.action_num)
        self.projections = np.random.randn(self.hash_size, inp_dimensions)

    def generate_hash(self, inp_vector):
        bools = (np.dot(inp_vector, self.projections.T) > 0).astype('int')
        return np.array([int(''.join(item.astype('str')), 2) for item in bools])

    def set_item(self, inp_vec, action_box):
        hash_value = self.generate_hash(inp_vec)
        for key, item in self.hash_table.items():
            index_1 = hash_value == key
            action_box_subset = list(action_box[index_1])
            c = Counter(action_box_subset)
            for act in range(self.action_num):
                self.hash_table[key][act] += c[act]

    def check_counts(self, inp_vec):
        hash_value = self.generate_hash(inp_vec)
        action_counts_box = np.zeros([hash_value.shape[0], self.action_num])
        for key, item in self.hash_table.items():
            _index = hash_value == key
            action_counts_box[_index, :] = item
        return action_counts_box
