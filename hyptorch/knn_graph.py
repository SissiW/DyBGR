import torch
import numpy as np

def gen_adj_0NN(height, width):
    adj = np.zeros((height * width, height * width), dtype=int)
    adj = adj + np.eye(adj.shape[0])
    return adj

def gen_adj_8NN(height, width):
    m = np.arange(0, height * width).reshape(height, width)
    adj = np.zeros((height * width, height * width), dtype=int)
    for i in range(height - 1):
        for j in range(width - 1):
            adj[m[i, j], m[i, j + 1]] = adj[m[i, j + 1], m[i, j]] = 1
            adj[m[i, j], m[i + 1, j]] = adj[m[i + 1, j], m[i, j]] = 1
            adj[m[i, j], m[i + 1, j + 1]] = adj[m[i + 1, j + 1], m[i, j]] = 1

            adj[m[i, j + 1], m[i + 1, j + 1]] = adj[m[i + 1, j + 1], m[i, j + 1]] = 1
            adj[m[i + 1, j], m[i + 1, j + 1]] = adj[m[i + 1, j + 1], m[i + 1, j]] = 1
            adj[m[i + 1, j], m[i, j + 1]] = adj[m[i, j + 1], m[i + 1, j]] = 1
    adj = adj + np.eye(adj.shape[0])
    return adj

def gen_adj_16NN(height, width):
    m = np.arange(0, height * width).reshape(height, width)
    adj = np.zeros((height * width, height * width), dtype=int)
    for i in range(height - 2):
        for j in range(width - 2):
            adj[m[i, j], m[i, j+1]] = adj[m[i, j+1], m[i, j]] = 1
            adj[m[i, j], m[i, j+2]] = adj[m[i, j+2], m[i, j]] = 1
            adj[m[i, j+1], m[i, j+2]] = adj[m[i, j+2], m[i, j+1]] = 1

            adj[m[i, j], m[i+1, j]] = adj[m[i+1, j], m[i, j]] = 1
            adj[m[i, j], m[i+2, j]] = adj[m[i+2, j], m[i, j]] = 1
            adj[m[i+1, j], m[i+2, j]] = adj[m[i+2, j], m[i+1, j]] = 1

            adj[m[i, j], m[i+1, j+1]] = adj[m[i+1, j+1], m[i, j]] = 1
            adj[m[i, j], m[i+2, j+2]] = adj[m[i+2, j+2], m[i, j]] = 1
            adj[m[i+1, j+1], m[i+2, j+2]] = adj[m[i+2, j+2], m[i+1, j+1]] = 1

            adj[m[i+1, j], m[i+1, j+1]] = adj[m[i+1, j+1], m[i+1, j]] = 1
            adj[m[i, j+1], m[i+1, j+1]] = adj[m[i+1, j+1], m[i, j+1]] = 1
            adj[m[i, j+1], m[i+1, j]] = adj[m[i+1, j], m[i, j+1]] = 1

            adj[m[i+2, j], m[i+1, j+1]] = adj[m[i+1, j+1], m[i+2, j]] = 1
            adj[m[i+2, j], m[i+2, j+2]] = adj[m[i+2, j+2], m[i+2, j]] = 1
            adj[m[i+2, j], m[i, j+2]] = adj[m[i, j+2], m[i+2, j]] = 1

            adj[m[i+1, j+1], m[i, j+2]] = adj[m[i, j+2], m[i+1, j+1]] = 1
            adj[m[i+2, j+2], m[i, j+2]] = adj[m[i, j+2], m[i+2, j+2]] = 1

            if i == height - 3 or j == width - 3:
                adj[m[i, j+1], m[i+1, j+2]] = adj[m[i+1, j+2], m[i, j+1]] = 1
                adj[m[i, j+2], m[i+1, j+2]] = adj[m[i+1, j+2], m[i, j+2]] = 1
                adj[m[i+1, j+1], m[i+1, j+2]] = adj[m[i+1, j+2], m[i+1, j+1]] = 1

                adj[m[i + 1, j + 1], m[i + 2, j + 1]] = adj[m[i + 2, j + 1], m[i + 1, j + 1]] = 1
                adj[m[i + 1, j], m[i + 2, j + 1]] = adj[m[i + 2, j + 1], m[i + 1, j]] = 1
                adj[m[i + 2, j], m[i + 2, j + 1]] = adj[m[i + 2, j + 1], m[i + 2, j]] = 1

                adj[m[i + 2, j + 1], m[i + 1, j + 2]] = adj[m[i + 1, j + 2], m[i + 2, j + 1]] = 1
                adj[m[i + 2, j + 1], m[i + 2, j + 2]] = adj[m[i + 2, j + 2], m[i + 2, j + 1]] = 1
                adj[m[i + 1, j + 2], m[i + 2, j + 2]] = adj[m[i + 2, j + 2], m[i + 1, j + 2]] = 1

    adj = adj + np.eye(adj.shape[0])
    return adj

def Adj_Normalize(A):
    '''
    :param S: (B, N, N), a similar matrix
    :param knng: K-nearnest-neighbor relationship graph
    Aij = Sij when Vj in KNN(Vi), else 0
    :return: the row-normalize adj (D^-1 * A)
    '''
    D = torch.pow(A.sum(2), -1)
    D = torch.where(torch.isnan(D), torch.zeros_like(D), D).diag_embed()
    out = torch.bmm(D, A)
    return out

if __name__ == '__main__':
    adj = gen_adj_16NN(5, 5)
    print(adj.shape)
    print(adj)