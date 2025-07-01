import numpy as np
import torch.nn as nn
import torch

class HWGATEParams():
    def __init__(self, dataset_params, input_dim, device=None) -> None:
        self.kp_dim=input_dim
        self.num_kps=64
        self.temporal_dim=dataset_params['src_len']
        self.num_classes=dataset_params['num_class']
        self.embed_dim=128
        self.temporal_patch_size=2
        self.pe=True
        self.depths=[4, 4, 8]
        self.num_heads=[2, 4, 8]
        self.window_size=16
        self.drop_rate=0.1
        self.attn_drop_rate=0.0
        self.ff_ratio=2.
        self.norm_layer=nn.LayerNorm
        self.device=device

        self.edges = [[
                    [0, 1],
                    [0, 2],
                    [0, 3],
                    [3, 4],
                    [4, 5],
                    [5, 6],
                    [6, 7],
                    [6, 8],
                    [8, 9],
                    [8, 10],
                    [6, 10],
                    [10, 11],
                    [10, 12],
                    [6, 12],
                    [12, 13],
                    [12, 14],
                    [14, 15],
                    [6, 14],
                    [7, 9],
                    [9, 11],
                    [11, 13],
                    [13, 15],
                    [7, 15],
                    [7, 11],
                    [7, 13]

                ],
                [
                    [0, 1],
                    [0, 2],
                    [0, 3],
                    [3, 4],
                    [4, 5],
                    [5, 6],
                    [6, 7],
                    [6, 8],
                    [8, 9],
                    [8, 10],
                    [6, 10],
                    [10, 11],
                    [10, 12],
                    [6, 12],
                    [12, 13],
                    [12, 14],
                    [14, 15],
                    [6, 14],
                    [7, 9],
                    [9, 11],
                    [11, 13],
                    [13, 15],
                    [7, 15],
                    [7, 11],
                    [7, 13]
                ],
                [
                    [0, 1],
                    [0, 2],
                    [0, 3],
                    [3, 4],
                    [4, 5],
                    [5, 6],
                    [6, 7],
                    [6, 8],
                    [8, 9],
                    [8, 10],
                    [6, 10],
                    [10, 11],
                    [10, 12],
                    [6, 12],
                    [12, 13],
                    [12, 14],
                    [14, 15],
                    [6, 14],
                    [7, 9],
                    [9, 11],
                    [11, 13],
                    [13, 15],
                    [7, 15],
                    [7, 11],
                    [7, 13]
                ],
                [
                    [0, 1],
                    [0, 2],
                    [0, 3],
                    [3, 4],
                    [4, 5],
                    [5, 6],
                    [6, 7],
                    [6, 8],
                    [8, 9],
                    [8, 10],
                    [6, 10],
                    [10, 11],
                    [10, 12],
                    [6, 12],
                    [12, 13],
                    [12, 14],
                    [14, 15],
                    [6, 14],
                    [7, 9],
                    [9, 11],
                    [11, 13],
                    [13, 15],
                    [7, 15],
                    [7, 11],
                    [7, 13]
                ],]

        self.adj_mat = torch.tensor(self.get_adj_mat(), dtype=torch.float32)
    
    def get_adj_mat(self):
        TP, W, K = self.temporal_patch_size, self.window_size, self.num_kps
        adj_mat_adj = [self.get_adj(i) for i in range(len(self.edges))]
        adj_mat = []
        for w in range(K//W):
            adj_mat_w = []
            for i in range(TP):
                adj_mat_r = []
                for j in range(TP):
                    if i==j:
                        adj_mat_r.append(adj_mat_adj[w])
                    elif abs(i-j) == 1.0:
                        adj_mat_r.append(np.eye(W))
                    else:
                        adj_mat_r.append(np.zeros((W, W)))
                adj_mat_w.append(np.concatenate(adj_mat_r, axis=1))
            adj_mat.append(np.concatenate(adj_mat_w))
        adj_mat = np.array(adj_mat)

        return adj_mat

    def get_adj(self, index):
        temp = np.eye(self.window_size)

        for i in self.edges[index]:
            temp[tuple(i)] = 1
            temp[tuple(i)[::-1]] = 1
        return temp
    
    def get_model_params(self):
        return self.kp_dim, self.num_kps, self.temporal_dim,self.num_classes,self.embed_dim,self.temporal_patch_size,self.pe,self.depths,self.num_heads,self.window_size,self.adj_mat,self.drop_rate,self.attn_drop_rate,self.ff_ratio,self.norm_layer,self.device
