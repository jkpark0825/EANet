import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

class Interaction_score(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.channels = dim
        self.norm = nn.LayerNorm(dim)
        
        self.q1 = nn.Linear(21, 1)
        self.q2 = nn.Linear(dim, dim)
        self.k1 = nn.Linear(21, 1)
        self.k2 = nn.Linear(dim, dim)
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.r_embedding = nn.Parameter(torch.randn(1, 1, 512))
        self.l_embedding = nn.Parameter(torch.randn(1, 1, 512))
    
    def forward(self, rhand_feat, lhand_feat):
        b, j, c = rhand_feat.shape
        rhand_feat = self.q1(rhand_feat.permute(0,2,1)).permute(0,2,1)
        rhand_feat = self.q2(self.norm(rhand_feat + self.r_embedding))
        rhand_feat = rhand_feat.reshape(b, 1, self.num_heads, c//self.num_heads).permute(0,2,1,3)
        lhand_feat = self.k1(lhand_feat.permute(0,2,1)).permute(0,2,1)
        lhand_feat = self.k2(self.norm(lhand_feat + self.l_embedding))
        lhand_feat = lhand_feat.reshape(b, 1, self.num_heads, c//self.num_heads).permute(0,2,1,3)
        
        attn = torch.matmul(rhand_feat, lhand_feat.transpose(-2,-1)) * self.scale
        attn = attn.sigmoid() # B head 1 1

        return attn
        
class Transformer(nn.Module):
    def __init__(self, dim=512, depth=2, num_heads=4, mlp_ratio=4.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio))

    def forward(self, query, key, interact=None):
        output = query
        for i, layer in enumerate(self.layers):
            output = layer(query=output, key=key, interact=interact)
        return output

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
    def forward(self, query, key, value, interact):
        B, N, C = query.shape
        query = query.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        key = key.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        value = value.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = torch.matmul(attn, value)
        if interact != None:
            x = (x*interact).transpose(1, 2).reshape(B, N, C)
        else:
            x = (x).transpose(1, 2).reshape(B, N, C)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()

        self.channels = dim
        self.norm = nn.LayerNorm(dim)
        self.encode_value = nn.Linear(dim, dim)
        self.encode_query = nn.Linear(dim, dim)
        self.encode_key = nn.Linear(dim, dim)

        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)
        
        self.q_embedding = nn.Parameter(torch.randn(1, 21, 512))
        self.k_embedding = nn.Parameter(torch.randn(1, 21, 512))
        self.v_embedding = nn.Parameter(torch.randn(1, 21, 512))
        
    def forward(self, query, key, interact = None):
        q_embed = self.norm(query + self.q_embedding)
        k_embed = self.norm(key + self.k_embedding)
        v_embed = self.norm(key + self.v_embedding)

        q = self.encode_query(q_embed)
        k = self.encode_key(k_embed)
        v = self.encode_value(v_embed)
        
        if interact == None:
            query = query + self.attn(query=q, key=k, value=v, interact = None)    
        else:
            query = query + self.attn(query=q, key=k, value=v, interact = interact)
        query = query + self.mlp(self.norm2(query))
        return query