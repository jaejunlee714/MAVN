import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Annotated
from collections import OrderedDict
import math

class GCNLayer_LRGB(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, act: str = "ReLU", dropout: float = 0.1, normalize = "None"):
        super(GCNLayer_LRGB, self).__init__()
        
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

        if normalize == "None":
            self.normalize = nn.Identity()
        elif normalize == "BatchNorm":
            self.normalize = nn.BatchNorm1d(dim_in)
        elif normalize == "LayerNorm":
            self.normalize = nn.LayerNorm(dim_in)
        else:
            raise NotImplementedError

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.W = nn.Linear(dim_in, dim_out, bias = True)
        self.drop = nn.Dropout(p = dropout)

        self.param_init()
    
    def param_init(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)

    def forward(self, reps_N: torch.FloatTensor, node_weights: torch.FloatTensor, edges: torch.LongTensor, edge_weights: torch.FloatTensor, reps_E: torch.FloatTensor):
        norm_reps_N = self.normalize(reps_N)
        num_N = len(reps_N)

        # +1 for self-loop
        sqrt_cnt_in_N = torch.sqrt(torch.zeros((num_N, )).cuda().index_add(dim = 0, index = edges[:,1], source = edge_weights) + node_weights)
        sqrt_cnt_out_N = torch.sqrt(torch.zeros((num_N, )).cuda().index_add(dim = 0, index = edges[:,0], source = edge_weights) + node_weights)
        edge_weights_GCN = 1/torch.index_select(sqrt_cnt_out_N, 0, edges[:,0]) * edge_weights * 1/torch.index_select(sqrt_cnt_in_N, 0, edges[:,1])

        msgs_N = self.W(norm_reps_N)

        aggr_N = torch.zeros((num_N, self.dim_out)).cuda().index_add(dim = 0, index = edges[:,1], source = edge_weights_GCN.unsqueeze(dim = 1) * torch.index_select(msgs_N, 0, edges[:,0]))

        reps_N  = reps_N + self.drop(self.act(msgs_N * node_weights.unsqueeze(dim = 1) * 1/(sqrt_cnt_in_N * sqrt_cnt_out_N).unsqueeze(dim=1) + aggr_N)) # add self-loop

        return reps_N, reps_E

class GINELayer_LRGB(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, act: str = "ReLU", dropout: float = 0.1, normalize = "None"):
        super(GINELayer_LRGB, self).__init__()
        
        if act == "ReLU":
            self.act = nn.ReLU
        elif act == "LeakyReLU":
            self.act = nn.LeakyReLU
        elif act == "PReLU":
            self.act = nn.PReLU
        elif act == "GELU":
            self.act = nn.GELU
        else:
            raise NotImplementedError
        
        if normalize == "None":
            self.normalize = nn.Identity()
        elif normalize == "BatchNorm":
            self.normalize = nn.BatchNorm1d(dim_in)
        elif normalize == "LayerNorm":
            self.normalize = nn.LayerNorm(dim_in)
        else:
            raise NotImplementedError

        self.message_ftn = nn.Sequential(OrderedDict([
                                          ("Lin1", nn.Linear(dim_in, dim_out, bias = True)),
                                          ("Act", self.act()),
                                          ("Drop1", nn.Dropout(p = dropout)),
                                          ("Lin2", nn.Linear(dim_out, dim_out, bias = True)),
                                          ("Drop2", nn.Dropout(p = dropout))]))
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.param_init()
    
    def param_init(self):   
        nn.init.kaiming_uniform_(self.message_ftn.Lin1.weight, a = math.sqrt(5))
        nn.init.uniform_(self.message_ftn.Lin1.bias, a = -1.0/math.sqrt(self.dim_in), b = 1.0/math.sqrt(self.dim_in))
        nn.init.kaiming_uniform_(self.message_ftn.Lin2.weight, a = math.sqrt(5))
        nn.init.uniform_(self.message_ftn.Lin2.bias, a = -1.0/math.sqrt(self.dim_out), b = 1.0/math.sqrt(self.dim_out))

    def forward(self, reps_N: torch.FloatTensor, node_weights: torch.FloatTensor, edges: torch.LongTensor, edge_weights: torch.FloatTensor, reps_E: torch.FloatTensor):
        norm_reps_N = self.normalize(reps_N)
        msg_N = reps_N.index_add(dim = 0, index = edges[:,1], source = F.relu(edge_weights.unsqueeze(dim = 1) * torch.index_select(norm_reps_N, 0, edges[:,0]) + reps_E))

        return reps_N + self.message_ftn(msg_N), reps_E

class GatedGCNLayer_LRGB(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, act: str = "ReLU", dropout: float = 0.1, normalize: str = "None"):
        super(GatedGCNLayer_LRGB, self).__init__()
        
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

        if normalize == "None":
            self.normalize = nn.Identity()
        elif normalize == "BatchNorm":
            self.normalize = nn.BatchNorm1d(dim_in)
        elif normalize == "LayerNorm":
            self.normalize = nn.LayerNorm(dim_in)
        else:
            raise NotImplementedError
        
        self.A = nn.Linear(dim_in, dim_out, bias = True)
        self.B = nn.Linear(dim_in, dim_out, bias = True)
        self.C = nn.Linear(dim_in, dim_out, bias = True)
        self.D = nn.Linear(dim_in, dim_out, bias = True)
        self.E = nn.Linear(dim_in, dim_out, bias = True)
        self.bn_N = nn.BatchNorm1d(dim_out)
        self.bn_E = nn.BatchNorm1d(dim_out)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.drop = nn.Dropout(p = dropout)
        self.param_init()
    
    def param_init(self):   
        nn.init.kaiming_uniform_(self.A.weight, a = math.sqrt(5))
        nn.init.uniform_(self.A.bias, a = -1.0/math.sqrt(self.dim_in), b = 1.0/math.sqrt(self.dim_in))
        nn.init.kaiming_uniform_(self.B.weight, a = math.sqrt(5))
        nn.init.uniform_(self.B.bias, a = -1.0/math.sqrt(self.dim_in), b = 1.0/math.sqrt(self.dim_in))
        nn.init.kaiming_uniform_(self.C.weight, a = math.sqrt(5))
        nn.init.uniform_(self.C.bias, a = -1.0/math.sqrt(self.dim_in), b = 1.0/math.sqrt(self.dim_in))
        nn.init.kaiming_uniform_(self.D.weight, a = math.sqrt(5))
        nn.init.uniform_(self.D.bias, a = -1.0/math.sqrt(self.dim_in), b = 1.0/math.sqrt(self.dim_in))
        nn.init.kaiming_uniform_(self.E.weight, a = math.sqrt(5))
        nn.init.uniform_(self.E.bias, a = -1.0/math.sqrt(self.dim_in), b = 1.0/math.sqrt(self.dim_in))

    def forward(self, reps_N: torch.FloatTensor, node_weights: torch.FloatTensor, edges: torch.LongTensor, edge_weights: torch.FloatTensor, reps_E: torch.FloatTensor):
        # GatedGCNLayer
        # Same with GraphGPS (https://github.com/rampasek/GraphGPS/blob/main/graphgps/layer/gatedgcn_layer.py)
        # Same with DGL (https://docs.dgl.ai/en/2.0.x/generated/dgl.nn.pytorch.conv.GatedGCNConv.html)

        norm_reps_N = self.normalize(reps_N)
        num_N = len(reps_N)
        BN = self.B(norm_reps_N)
        DN = self.D(norm_reps_N)
        EN = self.E(norm_reps_N)
        freps_E = torch.index_select(DN, 0, edges[:,0]) + torch.index_select(EN, 0, edges[:,1]) + self.C(reps_E)
        sreps_E = edge_weights.unsqueeze(dim = 1) * F.sigmoid(freps_E)
            
        in_N = torch.zeros_like(reps_N).index_add(dim = 0, index = edges[:,1], source = sreps_E)
        freps_N = self.A(norm_reps_N).index_add(dim = 0, index = edges[:,1], source = sreps_E  / (torch.index_select(in_N, 0, edges[:,1]) + 1e-6) * torch.index_select(BN, 0, edges[:,0]))

        return reps_N + self.drop(self.act(self.bn_N(freps_N))),  reps_E + self.drop(self.act(self.bn_E(freps_E)))

class GCNLayer_TunedGNN(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, act: str = "ReLU", dropout: float = 0.1, normalize: str = "None", res: bool = True):
        super(GCNLayer_TunedGNN, self).__init__()
        
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

        self.normalize_type = normalize
        if normalize == "None":
            self.normalize = nn.Identity()
        elif normalize == "BatchNorm":
            self.normalize = nn.BatchNorm1d(dim_out)
        elif normalize == "LayerNorm":
            self.normalize = nn.LayerNorm(dim_out)
        else:
            raise NotImplementedError

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.res = res
        self.W = nn.Linear(dim_in, dim_out, bias = False)
        self.bias = nn.Parameter(torch.empty(1, dim_out))
        self.drop = nn.Dropout(p = dropout)
        if res:
            self.W2 = nn.Linear(dim_in, dim_out, bias = True)

        self.param_init()
    
    def param_init(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.bias)
        if self.res:
            self.W2.reset_parameters()
        if self.normalize_type != "None":
            self.normalize.reset_parameters()

    def forward(self, reps_N: torch.FloatTensor, node_weights: torch.FloatTensor, edges: torch.LongTensor, edge_weights: torch.FloatTensor, reps_E: torch.FloatTensor):
        num_N = len(reps_N)

        # +1 for self-loop
        sqrt_cnt_in_N = torch.sqrt(torch.zeros((num_N, )).cuda().index_add(dim = 0, index = edges[:,1], source = edge_weights) + node_weights)
        sqrt_cnt_out_N = torch.sqrt(torch.zeros((num_N, )).cuda().index_add(dim = 0, index = edges[:,0], source = edge_weights) + node_weights)
        edge_weights_GCN = 1/torch.index_select(sqrt_cnt_out_N, 0, edges[:,0]) * edge_weights * 1/torch.index_select(sqrt_cnt_in_N, 0, edges[:,1])

        msgs_N = self.W(reps_N)

        aggr_N = torch.zeros((num_N, self.dim_out)).cuda().index_add(dim = 0, index = edges[:,1], source = edge_weights_GCN.unsqueeze(dim = 1) * torch.index_select(msgs_N, 0, edges[:,0]))

        if self.res:
            reps_N = self.drop(self.act(self.normalize(self.W2(reps_N)+msgs_N * node_weights.unsqueeze(dim = 1) * 1/(sqrt_cnt_in_N * sqrt_cnt_out_N).unsqueeze(dim=1) + aggr_N) + self.bias)) # add self-loop
        else:
            reps_N = self.drop(self.act(self.normalize(msgs_N * node_weights.unsqueeze(dim = 1) * 1/(sqrt_cnt_in_N * sqrt_cnt_out_N).unsqueeze(dim=1) + aggr_N) + self.bias))

        return reps_N, reps_E

class SAGELayer_TunedGNN(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, act: str = "ReLU", dropout: float = 0.1, normalize = "None", res:bool = True):
        super(SAGELayer_TunedGNN, self).__init__()
        
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

        self.normalize_type = normalize
        if normalize == "None":
            self.normalize = nn.Identity()
        elif normalize == "BatchNorm":
            self.normalize = nn.BatchNorm1d(dim_out)
        elif normalize == "LayerNorm":
            self.normalize = nn.LayerNorm(dim_out)
        else:
            raise NotImplementedError
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.res = res
        self.W = nn.Linear(dim_in, dim_out, bias = True)
        self.W_r = nn.Linear(dim_in, dim_out, bias = True)
        self.drop = nn.Dropout(dropout)
        if res:
            self.W2 = nn.Linear(dim_in, dim_out, bias = True)
        
        self.param_init()
        
    def param_init(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)
        nn.init.xavier_uniform_(self.W_r.weight)
        nn.init.zeros_(self.W_r.bias)
        if self.res:
            self.W2.reset_parameters()
        if self.normalize_type != "None":
            self.normalize.reset_parameters()

    def forward(self, reps_N: torch.FloatTensor, node_weights: torch.FloatTensor, edges: torch.LongTensor, edge_weights: torch.FloatTensor, reps_E: torch.FloatTensor):
        num_N = len(reps_N)

        aggr_N = torch.zeros((num_N, self.dim_in)).cuda().index_add(dim = 0, index = edges[:,1], source = edge_weights.unsqueeze(dim = 1) * torch.index_select(reps_N, 0, edges[:,0]))
        num_nbs = torch.zeros((num_N, 1)).cuda().index_add(dim = 0, index = edges[:,1], source = edge_weights.unsqueeze(dim = 1))
        aggr_N = aggr_N / torch.where(num_nbs == 0, 1, num_nbs)
        
        if self.res:
            reps_N = self.drop(self.act(self.normalize(self.W2(reps_N) + self.W_r(reps_N) + self.W(aggr_N))))
        else:
            reps_N = self.drop(self.act(self.normalize(self.W_r(reps_N) + self.W(aggr_N))))
        
        return reps_N, reps_E

class GATLayer_TunedGNN(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int = 1, act: str = "ReLU", dropout: float = 0.1, normalize:str = "None", res:bool = True):
        super(GATLayer_TunedGNN, self).__init__()
        
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
        
        if normalize == "None":
            self.normalize = nn.Identity()
        elif normalize == "BatchNorm":
            self.normalize = nn.BatchNorm1d(dim_out)
        elif normalize == "LayerNorm":
            self.normalize = nn.LayerNorm(dim_out)
        else:
            raise NotImplementedError
        
        self.num_heads = num_heads
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.res = res
        
        self.dim_head = dim_out // num_heads
        
        assert dim_out % num_heads == 0, "dim_out must be divisible by num_heads"
        
        self.W = nn.Linear(dim_in, dim_out, bias = False)
        self.attn_l = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, self.dim_head))
        )
        self.attn_r = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, self.dim_head))
        )
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        
        if res:
            self.W2 = nn.Linear(dim_in, dim_out, bias = True)
        self.drop = nn.Dropout(dropout)
        
        self.param_init()
    
    def param_init(self):
        nn.init.xavier_uniform_(self.W.weight)
        if self.res:
            self.W2.reset_parameters()
        
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)
    
    def forward(self, reps_N: torch.FloatTensor, node_weights: torch.FloatTensor, edges: torch.LongTensor, edge_weights: torch.FloatTensor, reps_E: torch.FloatTensor):
        num_N = len(reps_N)

        msgs_N = self.W(reps_N)
        h_src = h_dst = msgs_N.view(-1, self.num_heads, self.dim_head)
        
        src = edges[:,0]
        dst = edges[:,1]
        edge_weights = edge_weights.unsqueeze(dim = 1).unsqueeze(dim = 2)
        e_l = (h_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        e_r = (h_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        e = self.leaky_relu(torch.index_select(e_l, 0, src) + torch.index_select(e_r, 0, dst))
        
        e_max = torch.zeros((num_N, self.num_heads, 1)).cuda().index_reduce(dim = 0, index = dst, reduce="amax", include_self=False, source = e.detach())
        e = e - torch.index_select(e_max, 0, dst) # Stabilization
        e = e.exp()
        assert e.isnan().sum() == 0, "e is NaN"
        e_denoms = torch.zeros((num_N, self.num_heads, 1)).cuda().index_add(dim = 0, index = dst, source = edge_weights * e)
        
        aggr_N = torch.zeros((num_N, self.num_heads, self.dim_head)).cuda().index_add(
            dim = 0, index = dst, source = torch.index_select(h_src, 0, src) * e * edge_weights) / (e_denoms+1e-6)
        
        if self.res:
            reps_N = self.drop(self.act(self.normalize(self.W2(reps_N) + aggr_N.view(-1, self.dim_out))))
        else:
            reps_N = self.drop(self.act(self.normalize(aggr_N.view(-1, self.dim_out))))
        
        return reps_N, reps_E

class GCNLayer_Hete(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, act: str = "ReLU", dropout: float = 0.1, normalize = "None"):
        super(GCNLayer_Hete, self).__init__()
        
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

        if normalize == "None":
            self.normalize = nn.Identity()
        elif normalize == "BatchNorm":
            self.normalize = nn.BatchNorm1d(dim_in)
        elif normalize == "LayerNorm":
            self.normalize = nn.LayerNorm(dim_in)
        else:
            raise NotImplementedError

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.W = nn.Linear(dim_in, dim_out, bias = True)
        self.drop = nn.Dropout(p = dropout)
        self.W2 = nn.Linear(dim_out, dim_out, bias = True)

        self.param_init()
    
    def param_init(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.zeros_(self.W2.bias)

    def forward(self, reps_N: torch.FloatTensor, node_weights: torch.FloatTensor, edges: torch.LongTensor, edge_weights: torch.FloatTensor, reps_E: torch.FloatTensor):
        norm_reps_N = self.normalize(reps_N)
        num_N = len(reps_N)

        # +1 for self-loop
        sqrt_cnt_in_N = torch.sqrt(torch.zeros((num_N, )).cuda().index_add(dim = 0, index = edges[:,1], source = edge_weights) + node_weights)
        sqrt_cnt_out_N = torch.sqrt(torch.zeros((num_N, )).cuda().index_add(dim = 0, index = edges[:,0], source = edge_weights) + node_weights)
        edge_weights_GCN = 1/torch.index_select(sqrt_cnt_out_N, 0, edges[:,0]) * edge_weights * 1/torch.index_select(sqrt_cnt_in_N, 0, edges[:,1])

        msgs_N = self.W(norm_reps_N)

        aggr_N = torch.zeros((num_N, self.dim_out)).cuda().index_add(dim = 0, index = edges[:,1], source = edge_weights_GCN.unsqueeze(dim = 1) * torch.index_select(msgs_N, 0, edges[:,0]))

        reps_N  = reps_N + self.drop(self.W2(self.act(self.drop(msgs_N * node_weights.unsqueeze(dim = 1) * 1/(sqrt_cnt_in_N * sqrt_cnt_out_N).unsqueeze(dim=1) + aggr_N)))) # add self-loop

        return reps_N, reps_E

class SAGELayer_Hete(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, act: str = "ReLU", dropout: float = 0.1, normalize = "None"):
        super(SAGELayer_Hete, self).__init__()
        
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
        
        if normalize == "None":
            self.normalize = nn.Identity()
        elif normalize == "BatchNorm":
            self.normalize = nn.BatchNorm1d(dim_in)
        elif normalize == "LayerNorm":
            self.normalize = nn.LayerNorm(dim_in)
        else:
            raise NotImplementedError
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.W = nn.Linear(dim_in * 2, dim_out, bias = True)
        self.drop = nn.Dropout(dropout)
        self.W2 = nn.Linear(dim_out, dim_out, bias = True)
        
        self.param_init()
        
    def param_init(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.zeros_(self.W2.bias)
        
    def forward(self, reps_N: torch.FloatTensor, node_weights: torch.FloatTensor, edges: torch.LongTensor, edge_weights: torch.FloatTensor, reps_E: torch.FloatTensor):
        norm_reps_N = self.normalize(reps_N)
        num_N = len(reps_N)

        aggr_N = torch.zeros((num_N, self.dim_out)).cuda().index_add(dim = 0, index = edges[:,1], source = edge_weights.unsqueeze(dim = 1) * torch.index_select(norm_reps_N, 0, edges[:,0]))
        num_nbs = torch.zeros((num_N, 1)).cuda().index_add(dim = 0, index = edges[:,1], source = edge_weights.unsqueeze(dim = 1))
        aggr_N = aggr_N / torch.where(num_nbs == 0, 1, num_nbs)
        
        reps_N_new = torch.cat([aggr_N, norm_reps_N], dim = -1)
        reps_N_new = self.drop(self.W2(self.act(self.drop(self.W(reps_N_new)))))
        
        # Residual connection
        reps_N  = reps_N + reps_N_new
        
        return reps_N, reps_E

class GATLayer_Hete(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int = 8, act: str = "ReLU", dropout: float = 0.1, normalize = "None"):
        super(GATLayer_Hete, self).__init__()
        
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
        
        if normalize == "None":
            self.normalize = nn.Identity()
        elif normalize == "BatchNorm":
            self.normalize = nn.BatchNorm1d(dim_in)
        elif normalize == "LayerNorm":
            self.normalize = nn.LayerNorm(dim_in)
        else:
            raise NotImplementedError
        
        self.num_heads = num_heads
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        self.dim_head = dim_out // num_heads
        
        assert dim_out % num_heads == 0, "dim_out must be divisible by num_heads"
        
        self.W = nn.Linear(dim_in, dim_out, bias = True)
        self.attn_l = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, self.dim_head))
        )
        self.attn_r = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, self.dim_head))
        )
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        
        self.W2 = nn.Linear(dim_out + dim_in, dim_out, bias = True)
        self.drop = nn.Dropout(dropout)
        
        self.param_init()
    
    def param_init(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.zeros_(self.W2.bias)
        
        nn.init.xavier_normal_(self.attn_l)
        nn.init.xavier_normal_(self.attn_r)
    
    def forward(self, reps_N: torch.FloatTensor, node_weights: torch.FloatTensor, edges: torch.LongTensor, edge_weights: torch.FloatTensor, reps_E: torch.FloatTensor):
        norm_reps_N = self.normalize(reps_N)
        num_N = len(reps_N)

        msgs_N = self.W(norm_reps_N)
        h_src = h_dst = msgs_N.view(-1, self.num_heads, self.dim_head)
        
        # Compute e value for each edge in edges
        src = edges[:,0]
        dst = edges[:,1]
        edge_weights = edge_weights.unsqueeze(dim = 1).unsqueeze(dim = 2)
        e_l = (h_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        e_r = (h_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        e = self.leaky_relu(torch.index_select(e_l, 0, src) + torch.index_select(e_r, 0, dst))
        
        # Edge softmax (same dst means same group)
        e_max = torch.zeros((num_N, self.num_heads, 1)).cuda().index_reduce(dim = 0, index = dst, reduce="amax", include_self=False, source = e.detach())
        e = e - torch.index_select(e_max, 0, dst) # Stabilization
        e = e.exp()
        assert e.isnan().sum() == 0, "e is NaN"
        e_denoms = torch.zeros((num_N, self.num_heads, 1)).cuda().index_add(dim = 0, index = dst, source = edge_weights * e)
        
        # Aggregation based on attention coefficients
        aggr_N = torch.zeros((num_N, self.num_heads, self.dim_head)).cuda().index_add(
            dim = 0, index = dst, source = torch.index_select(h_src, 0, src) * e * edge_weights) / (e_denoms+1e-6)
        
        # Aggregation of multiple heads (num_nodes, num_heads, dim_head) -> (num_nodes, dim_out)
        reps_N_new = torch.cat([aggr_N.view(-1, self.dim_out), norm_reps_N], dim = 1)
        reps_N_new = self.act(self.drop(self.W2(reps_N_new)))
        
        # Residual connection
        reps_N  = reps_N + reps_N_new
        
        return reps_N, reps_E