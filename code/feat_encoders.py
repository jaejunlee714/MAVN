import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Annotated
from collections import OrderedDict
import math

class CategoricalEncoder(nn.Module):
    def __init__(self, num_feats:int, num_cats_per_feat: List[int], dim_out: int):
        super(CategoricalEncoder, self).__init__()

        self.num_feats = num_feats
        self.dim_out = dim_out

        feat_emb_list = []

        for i in range(num_feats):
            feat_emb_list.append(nn.Parameter(torch.zeros(num_cats_per_feat[i], dim_out)))

        self.feat_emb_list = nn.ParameterList(feat_emb_list)
        
        self.param_init()
    
    def param_init(self):
        for i in range(self.num_feats):
            nn.init.xavier_uniform_(self.feat_emb_list[i])
    
    def forward(self, feats):
        new_feats = torch.zeros(len(feats), self.dim_out).cuda()
        for i in range(self.num_feats):
            new_feats = new_feats + torch.index_select(self.feat_emb_list[i], 0, feats[:,i])
        return new_feats

class LinearEncoder(nn.Module):
    def __init__(self, num_feats:int, dim_out: int):
        super(LinearEncoder, self).__init__()

        self.num_feats = num_feats
        self.dim_out = dim_out

        self.layer = nn.Linear(num_feats, dim_out, bias = True)
        
        self.param_init()
    
    def param_init(self):
        nn.init.kaiming_uniform_(self.layer.weight, a = math.sqrt(5))
        nn.init.uniform_(self.layer.bias, a = -1.0/math.sqrt(self.num_feats), b = 1.0/math.sqrt(self.num_feats))
    
    def forward(self, feats):
        return self.layer(feats)

class LinearDropEncoder(nn.Module):
    def __init__(self, num_feats:int, dim_out: int, dropout: float = 0.1):
        super(LinearDropEncoder, self).__init__()

        self.num_feats = num_feats
        self.dim_out = dim_out

        self.layer = nn.Linear(num_feats, dim_out, bias = True)
        self.drop = nn.Dropout(p = dropout)
        
        self.param_init()
    
    def param_init(self):
        self.layer.reset_parameters()
    
    def forward(self, feats):
        return self.drop(self.layer(feats))

class NonLinearEncoder(nn.Module):
    def __init__(self, num_feats:int, dim_out: int, dropout: float = 0.1, act: str = "GELU"):
        super(NonLinearEncoder, self).__init__()

        if act == "ReLU":
            self.act = nn.ReLU()
        elif act == "LeakyReLU":
            self.act = nn.LeakyReLU()
        elif act == "PReLU":
            self.act = nn.PReLU()
        elif act == "GELU":
            self.act = nn.GELU()
        else:
            raise NotImplementedError

        self.num_feats = num_feats
        self.dim_out = dim_out

        self.layer = nn.Linear(num_feats, dim_out, bias = True)
        self.drop = nn.Dropout(p = dropout)
        
        self.param_init()
    
    def param_init(self):
        nn.init.kaiming_uniform_(self.layer.weight, a = math.sqrt(5))
        nn.init.uniform_(self.layer.bias, a = -1.0/math.sqrt(self.num_feats), b = 1.0/math.sqrt(self.num_feats))
    
    def forward(self, feats):
        return self.act(self.drop(self.layer(feats)))