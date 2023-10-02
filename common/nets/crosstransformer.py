from functools import partial

import torch
import torch.nn as nn
import numpy as np
from itertools import repeat
import collections.abc

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=False, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key):
        B, Nq, C = query.shape # B 21 512
        _, Nk, _ = key.shape
        q = self.q(query).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0,2,1,3) #B  head 21 c//head
        k = self.k(key).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        v = self.v(key).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0,2,1,3)

        attn = (q @ k.transpose(-2, -1)) * self.scale # B head 21 42
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C) #B head 21 512
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm3 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, q, k):
        q = q + self.attn(self.norm1(q), self.norm2(k))
        q = q + self.mlp(self.norm3(q))
        return q


class CrossTransformer(nn.Module):
    def __init__(self, in_chans1=512, in_chans2=512, depth=4, num_heads=4, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, 64, in_chans1))
        self.pos_embed2 = nn.Parameter(torch.randn(1, 1+2*64, in_chans2))
        self.blocks = nn.ModuleList([
            Block(in_chans1, num_heads, mlp_ratio, qkv_bias=False, norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, q, k):
        q = q + self.pos_embed
        k = k + self.pos_embed2
        for blk in self.blocks:
            q = blk(q,k)
        return q 
        