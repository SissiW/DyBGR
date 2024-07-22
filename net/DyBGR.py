import math
from functools import partial
from itertools import repeat
import numpy as np
from torch_scatter import scatter_add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._six import container_abcs
from hyptorch.utils import euclidean_dist, MaxminNorm

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GraphLearningProp(nn.Module):
    def __init__(self, cfg, dim):
        super().__init__()
        self.K = cfg.K
        self.T = cfg.T
        self.epsilon = cfg.epsilon
        self.lam = cfg.lam
        self.beta = cfg.beta

        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(dim)
        # self.proj = nn.Linear(dim, dim)

    def forward(self, x, L):  # torch.Size([1, B, C]), L: torch.Size([B, B]) normalized
        _, B, C = x.shape
        # computer euclidean distance and sort
        H = x.squeeze(0)  # torch.Size([B, C]) 
        F = H

        D1 = euclidean_dist(F, F) - 2*self.beta*L - 0.00001 * MaxminNorm(torch.mm(F, F.t()))  # torch.Size([B, B])
        D2 = MaxminNorm(D1)
        zero = torch.zeros(B, B).cuda()
        D = torch.where(D2>0, D2, zero)
        Dx, index = torch.sort(D) # torch.Size([B, B]) 

        # compute gamma
        d = Dx[:, 1:self.K+2]  # torch.Size([B, K+1])
        dk = d[:, self.K].view(B, 1)  # torch.Size([B, 1])
        dk_sum = torch.sum(Dx[:, 1:self.K+1], 1).view(B, 1)  # torch.Size([B, 1])
        gamma = torch.mean(0.5*(self.K*dk-dk_sum))  

        # compute eta
        eta = 1/self.K * (1 + Dx[:, 1:self.K+1].sum(dim=1, keepdim=True)/(2*gamma + 1e-8))  # torch.Size([B, 1])

        # I = torch.eye(B).cuda()
        # iteration: learning graph and update feature F
        for t in range(self.T):
            attn = MaxminNorm(torch.mm(F, F.t()))  # torch.Size([B, B])
            # S = torch.zeros_like(attn)  # initial similarity: torch.Size([B, B])

            # learning graph
            dis = torch.gather(D, dim=1, index=index[:, 1:self.K+1]) - self.lam* torch.gather(attn, dim=1, index=index[:, 1:self.K+1])  
            A = self.relu(eta.expand(B, self.K) - dis/(2*gamma + 1e-8))  # torch.Size([B, K])

            # updata feature: AX
            id_r = torch.arange(B).repeat(self.K, 1).t().reshape(1, -1).cuda()  # torch.Size([1, BK])
            id_c = index[:, 1:self.K+1].reshape(1, -1)
            edge = torch.cat([id_r, id_c], 0)
            A_value = A.reshape(-1)  # torch.Size(BK])

            out = scatter_add(H[edge[1]]*A_value.unsqueeze(1), edge[0], dim=0, dim_size=B)  # torch.Size([B, C])
            F = self.epsilon*out + (1-self.epsilon)*H  # torch.Size([B, C])
            
            
            '''S = S.scatter(dim=1, index=index[:, 1:self.K+1], src=A.half())  # torch.Size([B, B])
            # updata feature
            W = (S + self.epsilon*I).half()  # torch.Size([B, B]) 
            # np.savetxt('W.txt', W.detach().cpu().numpy()) 
            F = torch.mm(W, F)  # torch.Size([B, C])'''

        x = F.unsqueeze(0)         

        return x


class Block(nn.Module):

    def __init__(self, cfg, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.glp = GraphLearningProp(cfg, dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, label):  # torch.Size([1, B, C])
        x = x + self.drop_path(self.glp(self.norm1(x), self.label_graph(label)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # torch.Size([1, B, C])
        return x

    def adj_normalization(self, A):
        '''
        A: [N, N]
        '''
        D = torch.pow(A.sum(1).float(), -1)  
        # D = torch.pow(A.sum(1).float(), -5)
        D = torch.diag(D)
        adj = torch.matmul(D, A)  
        # adj = torch.matmul(torch.matmul(A, D).t(), D)
        return adj
    
    def label_graph(self, label):
        '''
        label: torch.Size([B])
        '''    
        B = label.size(0)  # B

        # construct semantic graph
        l1 = label.unsqueeze(1).repeat(1, B)  # torch.Size([B, B])
        l2 = label.t().repeat(B, 1)  # torch.Size([B, B])
        adj = l1.eq_(l2)  # torch.Size([B, B])

        # normalization 
        adj_loop = adj + torch.eye(B, B)
        A = self.adj_normalization(adj_loop).cuda()  # torch.Size([B, B])
        return A


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x