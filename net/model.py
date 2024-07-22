import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.factory import create_model
import hyptorch.nn as hypnn
from thop import profile
from .DyBGR import *


def init_model(cfg):
    # cfg = cfg.as_dict()
    if cfg.model.startswith("dino"):
        body = torch.hub.load("facebookresearch/dino:main", cfg.model)
    else:
        # body = timm.create_model(cfg.model, pretrained=True)
        body = create_model(cfg.model, pretrained=True)
    # if cfg.get("hyp_c", 0) > 0:
    if cfg.hyp_c > 0:
        last = hypnn.ToPoincare(
            c=cfg.hyp_c,
            ball_dim=cfg.emb,
            riemannian=False,
            clip_r=cfg.clip_r,
        )
    else:
        last = NormLayer()
    bdim = 2048 if cfg.model == "resnet50" else 768 #ViT-B:768 384
    head = nn.Sequential(nn.Linear(bdim, cfg.emb), last)
    nn.init.constant_(head[0].bias.data, 0)
    nn.init.orthogonal_(head[0].weight.data)
    rm_head(body)
    # if cfg.freeze is not None:
    #     freeze(body, cfg.freeze)
    model = HeadSwitch(body, head, cfg)
    model.cuda().train()
    return model


class HeadSwitch(nn.Module):
    def __init__(self, body, head, cfg):  
        super(HeadSwitch, self).__init__()
        self.depth = cfg.depth
        self.num_heads = cfg.num_heads

        self.body = body
        self.head = head
        self.norm = NormLayer()
        self.blocks = nn.ModuleList([Block(cfg, dim=768, num_heads=self.num_heads, mlp_ratio=4, qkv_bias=True, qk_scale=None, \
                                                    drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm) for i in
                                    range(self.depth)])

    def forward(self, x, label, skip_head=False):  
        '''
        x: torch.Size([414, 3, 224, 224])
        skip_head: False
        '''
        # label = torch.randn(405)
        # ViT
        x = self.body(x)  # torch.Size([414, 384])  torch.Size([459, 768])
        pred_x = x

        # BatchFormer
        if self.training:
            x = x.unsqueeze(0)  # torch.Size([1, 414, 384])
            for layer in self.blocks:
                x = layer(x, label)
            x = x.squeeze(0)  # torch.Size([414, 384])
            x = torch.cat([pred_x, x], 0)

        if type(x) == tuple:  # false
            x = x[0]
        if not skip_head:
            x = self.head(x)  # torch.Size([414*2, 128])  torch.Size([810, 768])
        else:
            x = self.norm(x)
        return x

class NormLayer(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)


def freeze(model, num_block):
    def fr(m):
        for param in m.parameters():
            param.requires_grad = False

    # fr(model.patch_embed)
    # fr(model.pos_drop)
    for i in range(num_block):
        fr(model.blocks[i])


def rm_head(m):
    names = set(x[0] for x in m.named_children())
    target = {"head", "fc", "head_dist"}
    for x in names & target:
        m.add_module(x, nn.Identity())
