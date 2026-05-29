import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Annotated
from collections import OrderedDict
from layers import *
from feat_encoders import *
from model_utils import *
import math

class GNN(nn.Module):
    def __init__(self, dim_in: Union[int, List[int]], dim_out: Union[int, List[int]], dim_mlp: Union[int, List[int]], dim_pe: int,
                    task: Annotated[str, "Graph Classification", "Graph Regression", "Node Classification"],
                    num_pe:int, num_class: int = 1,
                    num_layer: int = 6, num_pred_mlp_layer: int = 3,
                    act: str = "ReLU", dropout: float = 0.1, normalize: Annotated[str, "None", 'BatchNorm', "LayerNorm"] = "None", base: Annotated[str, "GCN", "GINE"] = "GCN",
                    readout: Annotated[str, "mean", "sum"] = "mean", coeff: float = 1.0,
                    N_feat_type: Annotated[str, "Categorical"] = "Categorical", num_N_feat: int = 10, N_feat_cats: List[int] = [],
                    E_feat_type: Annotated[str, "Categorical"] = "Categorical", num_E_feat: int = 3, E_feat_cats: List[int] = [],
                    pe_type: Annotated[str, "None", "RWSE", "LapPE"] = "None",
                    pe_norm: Annotated[str, "None", "Batch"] = "Batch", pred_mlp_layer_init: Annotated[str, "xavier_uniform", "xavier_normal", "default"] = "xavier_normal", 
                    normalize_before_pred: bool = True, res:bool = True, **kwargs):
        super(GNN, self).__init__()
        self.task = task
        self.readout = readout
        self.pred_mlp_layer_init = pred_mlp_layer_init

        layer_list = []

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
        if base == "GCN-LRGB":
            base_gnn = GCNLayer_LRGB
        elif base == "GINE-LRGB":
            base_gnn = GINELayer_LRGB
        elif base == "GatedGCN-LRGB":
            base_gnn = GatedGCNLayer_LRGB
        elif base == "GCN-TunedGNN":
            base_gnn = GCNLayer_TunedGNN
        elif base == "GraphSAGE-TunedGNN":
            base_gnn = SAGELayer_TunedGNN
        elif base == "GAT-TunedGNN":
            base_gnn = GATLayer_TunedGNN
        elif base == "GCN-Hete":
            base_gnn = GCNLayer_Hete
        elif base == "GraphSAGE-Hete":
            base_gnn = SAGELayer_Hete
        elif base == "GAT-Hete":
            base_gnn = GATLayer_Hete
        else:
            raise NotImplementedError
        
        if type(dim_in) is int:
            dim_in = [dim_in for _ in range(num_layer)]
            if N_feat_type == 'None':
                dim_in[0] = num_N_feat
        if type(dim_out) is int:
            dim_out = [dim_out for _ in range(num_layer)]

        if normalize == "None" or not normalize_before_pred:
            self.normalize = nn.Identity()
        elif normalize == "BatchNorm":
            self.normalize = nn.BatchNorm1d(dim_out[-1])
        elif normalize == "LayerNorm":
            self.normalize = nn.LayerNorm(dim_out[-1])
        else:
            raise NotImplementedError

        assert len(dim_in) == num_layer
        assert len(dim_out) == num_layer

        self.dim_in = dim_in
        
        self.dim_out = dim_out

        if type(dim_pe) == list:
            dim_pe_total = sum(dim_pe)
        else:
            dim_pe_total = dim_pe

        if N_feat_type == 'Categorical':
            self.enc_N = CategoricalEncoder(num_N_feat, N_feat_cats, dim_in[0] - dim_pe_total)
        elif N_feat_type == 'Linear':
            self.enc_N = LinearEncoder(num_N_feat, dim_in[0] - dim_pe_total)
        elif N_feat_type == 'NonLinear':
            self.enc_N = NonLinearEncoder(num_N_feat, dim_in[0] - dim_pe_total, dropout, act)
        elif N_feat_type == 'LinearDrop':
            self.enc_N = LinearDropEncoder(num_N_feat, dim_in[0] - dim_pe_total, dropout)
        elif N_feat_type == 'None':
            self.enc_N = nn.Identity()
        else:
            raise NotImplementedError
        
        if E_feat_type == 'Categorical':
            self.enc_E = CategoricalEncoder(num_E_feat, E_feat_cats, dim_in[0])
        elif E_feat_type == 'Linear':
            self.enc_E = LinearEncoder(num_E_feat, dim_in[0])
        elif E_feat_type == 'NonLinear':
            self.enc_E = NonLinearEncoder(num_E_feat, dim_in[0], dropout, act)
        elif E_feat_type == 'None':
            self.enc_E = nn.Identity()
        else:
            raise NotImplementedError
        
        self.dim_pe = dim_pe
        self.num_pe = num_pe
        self.pe_type = pe_type
        if pe_type == "RWSE":
            self.enc_pe = nn.Linear(num_pe, dim_pe, bias = True)
        elif pe_type == "LapPE":
            self.enc_pe = nn.Sequential(nn.Linear(2, 2*dim_pe),
                                        nn.ReLU(),
                                        nn.Linear(2*dim_pe, dim_pe),
                                        nn.ReLU())
        elif pe_type == "LapPE+RWSE":
            self.enc_pe1 = nn.Sequential(nn.Linear(2, 2*dim_pe[0]),
                                         nn.ReLU(),
                                         nn.Linear(2*dim_pe[0], dim_pe[0]),
                                         nn.ReLU())
            self.enc_pe2 = nn.Linear(num_pe[1], dim_pe[1], bias = True)
        elif pe_type == "None":
            self.enc_pe = nn.Identity()
        else:
            raise NotImplementedError

        if pe_norm == "Batch":
            self.pe_norm = nn.BatchNorm1d(num_pe)
        elif pe_norm == "None":
            self.pe_norm = nn.Identity()
        elif pe_norm == "Batch+None":
            self.pe_norm1 = nn.BatchNorm1d(num_pe[0])
            self.pe_norm2 = nn.Identity()
        elif pe_norm == "None+Batch":
            self.pe_norm1 = nn.Identity()
            self.pe_norm2 = nn.BatchNorm1d(num_pe[1])
        else:
            raise NotImplementedError

        for l in range(num_layer):
            if "TunedGNN" in base:
                layer_list.append(base_gnn(dim_in = dim_in[l], dim_out = dim_out[l], act = act, dropout = dropout, normalize = normalize, res = res))
            else:
                layer_list.append(base_gnn(dim_in = dim_in[l], dim_out = dim_out[l], act = act, dropout = dropout, normalize = normalize))
        self.num_layer = num_layer
        self.layers = nn.ModuleList(layer_list)

        if type(dim_mlp) is int:
            dim_mlp = [dim_mlp for _ in range(num_pred_mlp_layer - 1)]

        assert len(dim_mlp) == num_pred_mlp_layer - 1
        dim_mlp = [dim_out[-1]] + dim_mlp
        mlp_dict = OrderedDict([])
        if self.task in ["Graph Classification", "Graph Regression"]:
            for l in range(num_pred_mlp_layer-1):
                mlp_dict.update({f"drop{l+1}":nn.Dropout(p = dropout), \
                                 f"lin{l+1}":nn.Linear(dim_mlp[l], dim_mlp[l+1], bias = True), \
                                 f"act{l+1}":self.act()})
            mlp_dict.update({f"drop{num_pred_mlp_layer}":nn.Dropout(p = dropout)})
            if num_pred_mlp_layer > 0:
                mlp_dict.update({f"lin{num_pred_mlp_layer}":nn.Linear(dim_mlp[num_pred_mlp_layer-1], num_class, bias = True)})
        elif task == "Node Classification":
            for l in range(num_pred_mlp_layer-1):
                mlp_dict.update({f"lin{l+1}":nn.Linear(dim_mlp[l], dim_mlp[l+1], bias = True), \
                                 f"drop{l+1}":nn.Dropout(p = dropout), \
                                 f"act{l+1}":nn.ReLU()})
            if num_pred_mlp_layer > 0:
                mlp_dict.update({f"lin{num_pred_mlp_layer}":nn.Linear(dim_mlp[num_pred_mlp_layer-1], num_class, bias = True)})
        else:
            raise NotImplementedError
        self.num_pred_mlp_layer = num_pred_mlp_layer
        self.mlp = nn.Sequential(mlp_dict)
        self.num_class = num_class

        self.param_init()

    def param_init(self):
        for l in range(self.num_pred_mlp_layer):
            if self.task in ["Graph Classification", "Graph Regression"]:
                if self.pred_mlp_layer_init == "xavier_normal":
                    nn.init.xavier_normal_(self.mlp[3*l+1].weight, gain = nn.init.calculate_gain('relu'))
                    nn.init.zeros_(self.mlp[3*l+1].bias)
                elif self.pred_mlp_layer_init == "xavier_uniform":
                    nn.init.xavier_uniform_(self.mlp[3*l+1].weight)
                    nn.init.zeros_(self.mlp[3*l+1].bias)
                elif self.pred_mlp_layer_init == "default":
                    self.mlp[3*l+1].reset_parameters()
                else:
                    raise NotImplementedError
            elif self.task == "Node Classification":

                if self.pred_mlp_layer_init == "xavier_normal":
                    nn.init.xavier_normal_(self.mlp[3*l].weight, gain = nn.init.calculate_gain('relu'))
                    nn.init.zeros_(self.mlp[3*l].bias)
                elif self.pred_mlp_layer_init == "xavier_uniform":
                    nn.init.xavier_uniform_(self.mlp[3*l].weight)
                    nn.init.zeros_(self.mlp[3*l].bias)
                elif self.pred_mlp_layer_init == "default":
                    self.mlp[3*l].reset_parameters()
                
            else:
                raise NotImplementedError
        
        if self.pe_type == "RWSE":
            nn.init.kaiming_uniform_(self.enc_pe.weight, a = math.sqrt(5))
            nn.init.uniform_(self.enc_pe.bias, a = -1.0/math.sqrt(self.num_pe), b = 1.0/math.sqrt(self.num_pe))
        elif self.pe_type == "LapPE+RWSE":
            nn.init.kaiming_uniform_(self.enc_pe2.weight, a = math.sqrt(5))
            nn.init.uniform_(self.enc_pe2.bias, a = -1.0/math.sqrt(self.num_pe[1]), b = 1.0/math.sqrt(self.num_pe[1]))

    def forward(self, N2G: torch.LongTensor, E2G: torch.LongTensor, G2num_N: torch.LongTensor, 
                      feat_N: torch.FloatTensor, feat_E:torch.FloatTensor, E: torch.LongTensor, pe_feat_N: torch.FloatTensor, num_G: int, **kwargs):
        num_N = len(feat_N)
        num_E = len(E)
        if self.pe_type == "LapPE":
            if self.training:
                pe_feat_N[:,(torch.rand(self.num_pe) < 0.5),0] *= -1
                rep_pe = torch.sum(self.enc_pe(self.pe_norm(pe_feat_N)), dim = 1)
            else:
                rep_pe = torch.sum(self.enc_pe(self.pe_norm(pe_feat_N)), dim = 1)
        elif self.pe_type in ["None", "RWSE"]:
            rep_pe = self.enc_pe(self.pe_norm(pe_feat_N))
        elif self.pe_type == "LapPE+RWSE":
            lap = pe_feat_N[0]
            if self.training:
                lap[:,(torch.rand(self.num_pe[0]) < 0.5),0] *= -1
                rep_lap = torch.sum(self.enc_pe1(self.pe_norm1(lap)), dim = 1)
            else:
                rep_lap = torch.sum(self.enc_pe1(self.pe_norm1(lap)), dim = 1)
            rwse = pe_feat_N[1]
            rep_rwse = self.enc_pe2(self.pe_norm2(rwse))
            rep_pe = torch.cat([rep_lap, rep_rwse], dim = 1)
        else:
            raise NotImplementedError
        reps_N = torch.cat([self.enc_N(feat_N), rep_pe], dim = 1)
        node_weights = torch.ones((num_N, )).cuda()
        reps_E = self.enc_E(feat_E)
        edge_weights = torch.ones((len(E), )).cuda()

        for i, layer in enumerate(self.layers):
            reps_N, reps_E = layer(reps_N = reps_N, node_weights = node_weights, edges = E, edge_weights = edge_weights, reps_E = reps_E)
        
        if self.task in ["Graph Classification", "Graph Regression"]:
            if self.readout == "mean":
                reps_G = torch.zeros((len(G2num_N), self.dim_out[-1])).cuda().index_add(dim = 0, index = N2G, source = reps_N) / G2num_N.unsqueeze(dim = 1)
            elif self.readout == "sum":
                reps_G = torch.zeros((len(G2num_N), self.dim_out[-1])).cuda().index_add(dim = 0, index = N2G, source = reps_N)
            else:
                raise NotImplementedError
            pred = self.mlp(reps_G)
        elif self.task in ["Node Classification"]:
            pred = self.mlp(self.normalize(reps_N))
        else:
            raise NotImplementedError
    
        return pred, torch.zeros(self.num_layer,).cuda(), torch.zeros(self.num_layer,).cuda(), torch.zeros(self.num_layer, 0).cuda()

class MessagePathGenerator(nn.Module):
    def __init__(self, dim: int = 120, dim_dot: int = 120, dim_mlp:int = 120, num_head: int = 1, num_mlp_layer: int = 1, dropout:float = 0.1, act:str = "ReLU",  norm_N:str = "None", norm_VN:str = "None", aggr_VN:str = "normalized_sigmoid", num_VN:int = 12, log_w: float = 1.0):
        super(MessagePathGenerator, self).__init__()

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
        
        self.aggr_VN = aggr_VN

        self.norm_N = norm_N        
        if norm_N == "None":
            pass
        elif norm_N == "BatchNorm":
            self.norm_N_ftn = nn.BatchNorm1d(dim)
        elif norm_N == "LayerNorm":
            self.norm_N_ftn = nn.LayerNorm(dim)
        elif norm_N == "GraphNorm":
            self.a = nn.Parameter(torch.ones(1, dim))
            self.b = nn.Parameter(torch.zeros(1, dim))
            self.c = nn.Parameter(torch.ones(1, dim))
        else:
            raise NotImplementedError
        
        self.norm_VN = norm_VN
        if norm_VN == "None":
            pass
        elif norm_VN == "LayerNorm":
            self.norm_VN_ftn = nn.LayerNorm(dim_dot)
            self.norm_cVN_ftn = nn.LayerNorm(dim)
        else:
            raise NotImplementedError

        self.dim = dim
        self.num_head = num_head
        self.dim_per_head = dim_dot // num_head

        assert self.dim_per_head * num_head == dim_dot

        self.drop = nn.Dropout(p = dropout)

        self.num_mlp_layer = num_mlp_layer
        mlp_dict = OrderedDict([])
        if num_mlp_layer == 0:
            mlp_dict.update({"id":nn.Identity()})
        elif num_mlp_layer == 1:
            mlp_dict.update({f"lin0":nn.Linear(dim, dim_dot, bias = True)})
        else:
            mlp_dict.update({f"lin0":nn.Linear(dim, dim_mlp, bias = True),
                             f"drop0":nn.Dropout(p = dropout), \
                             f"act0":self.act()})
            for l in range(1, num_mlp_layer-1):
                mlp_dict.update({f"lin{l}":nn.Linear(dim_mlp, dim_mlp, bias = True), \
                                 f"drop{l}":nn.Dropout(p = dropout), \
                                 f"act{l}":self.act()})
            mlp_dict.update({f"lin{num_mlp_layer-1}":nn.Linear(dim_mlp, dim_dot, bias = True)})
            
        self.Q = nn.Sequential(mlp_dict)
        self.alpha = nn.Parameter(torch.zeros(2, self.num_head))
        self.beta = nn.Parameter(torch.zeros(2, self.dim))

        if log_w < 0:
            self.log_w = nn.Parameter(torch.zeros(1, self.num_head))
        else:
            self.log_w = log_w

        self.num_VN = num_VN
        self.kreps_VN = nn.Parameter(torch.zeros(num_VN, dim_dot))
        self.vreps_VN = nn.Parameter(torch.zeros(num_VN, dim))

        self.init_nE = nn.Parameter(torch.zeros(1, self.dim))
        self.init_nVN2VN = nn.Parameter(torch.zeros((1, self.dim)))
        self.param_init()

    def param_init(self):
        
        for l in range(self.num_mlp_layer):
            nn.init.kaiming_uniform_(self.Q[3*l].weight, a = math.sqrt(5))
            nn.init.uniform_(self.Q[3*l].bias, a = -1.0/math.sqrt(self.dim), b = 1.0/math.sqrt(self.dim))
    
        nn.init.xavier_normal_(self.init_nE, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.init_nVN2VN, gain = nn.init.calculate_gain('relu'))

        for idx in range(self.num_VN):
            nn.init.kaiming_uniform_(self.kreps_VN[idx].unsqueeze(dim = 0), a = math.sqrt(5))
            nn.init.kaiming_uniform_(self.vreps_VN[idx].unsqueeze(dim = 0), a = math.sqrt(5))


    def forward(self, gumbel_temperature:float, threshold: float,
                      reps_N: torch.FloatTensor, N_weights: torch.FloatTensor, G2rVN: torch.FloatTensor,
                      N2G: torch.LongTensor, G2num_N: torch.FloatTensor,
                      E: torch.LongTensor, E_weights: torch.FloatTensor, reps_E: torch.FloatTensor):
        

        device = reps_N.device
        num_G = len(G2num_N)
        num_N = len(reps_N)
        num_E = len(reps_E)

        G2VN_cands = G2rVN.nonzero()
        num_VN_cands = G2rVN.sum()

        if num_VN_cands == 0:
            return reps_N, N_weights, G2rVN, N2G, G2num_N, E, E_weights, reps_E

        ## Section 4.1
        ### Relevance Score Computation

        G2num_VN = G2rVN.sum(dim = -1)

        E_cands_N = torch.arange(num_N, device = device).repeat_interleave(G2num_VN[N2G].long())

        if self.norm_N in ["LayerNorm", "BatchNorm"]:
            norm_reps_N = self.norm_N_ftn(reps_N)
        elif self.norm_N in ["GraphNorm"]:
            norm_reps_N = GroupNormalize(cnt_groups = G2num_N, groups = N2G, vectors = reps_N, mean_weights = self.c) * self.a  + self.b
        elif self.norm_N in ["None"]:
            norm_reps_N = reps_N
        else:
            raise NotImplementedError

        VN_start_indices_per_G =  torch.cat([torch.zeros(1,device = device).long(), torch.cumsum(G2num_VN.long(), dim = 0)[:-1]], dim = 0)
        VN_start_indices_per_N = VN_start_indices_per_G[N2G]
        start_index_per_N = torch.cat([torch.zeros(1,device = device).long(), torch.cumsum(G2num_VN[N2G].long(), dim = 0)[:-1]], dim = 0)
        reiter_per_N = torch.arange((G2num_VN * G2num_N).sum().long(), device = device) - start_index_per_N.repeat_interleave(G2num_VN[N2G].long())
        E_cands_VN = reiter_per_N + VN_start_indices_per_N.repeat_interleave(G2num_VN[N2G].long())

        E_cands = torch.stack([E_cands_N, E_cands_VN], dim = 1)

        kreps_VN = torch.index_select(self.kreps_VN, 0, G2VN_cands[:,1])

        if self.norm_VN in ['LayerNorm']:
            kreps_VN = self.norm_VN_ftn(kreps_VN)
        elif self.norm_VN in ["None"]:
            kreps_VN = kreps_VN
        else:
            raise NotImplementedError

        E_cands_init_logit_per_head = (torch.index_select(self.Q(norm_reps_N), 0, E_cands[:, 0]) * torch.index_select(kreps_VN, 0, E_cands[:, 1])).view(-1, self.num_head, self.dim_per_head).sum(dim = -1) / math.sqrt(self.dim_per_head)

        ### VN Selection

        log_N = torch.log(G2num_N).unsqueeze(dim = 1)

        if type(self.log_w) == float:
            log_w = self.log_w
        elif type(self.log_w) == nn.parameter.Parameter:
            log_w = F.softplus(self.log_w)
        else:
            raise NotImplementedError

        E_cands_logits_Gwise_per_head = log_w * GroupLogSoftmax(num_groups = num_G, groups = N2G[E_cands[:,0]], logits = E_cands_init_logit_per_head) + E_cands_init_logit_per_head

        VN_cands_logits = GroupLogSumExp(num_groups = num_VN_cands, groups = E_cands[:,1], logits = E_cands_logits_Gwise_per_head.mean(dim = -1)) - torch.index_select(log_N, 0, G2VN_cands[:,0]).squeeze(dim = 1)

        if self.training:
            VN_cands_probs = GumbelSigmoid(VN_cands_logits, tau = gumbel_temperature)
        else:
            VN_cands_probs = F.sigmoid(VN_cands_logits)

        chosen_G2VN = torch.where(VN_cands_probs >= threshold, 1, 0) - VN_cands_probs.detach() + VN_cands_probs

        chosen_scores = chosen_G2VN[chosen_G2VN == 1]
        chosen_Gs = G2VN_cands[chosen_G2VN == 1, 0]

        num_cVN = chosen_G2VN.sum().long()

        G2rVN.masked_scatter_(G2rVN == 1, ~chosen_G2VN.detach().bool())

        print(f"avg VNs: {num_cVN/num_G}")

        if num_cVN == 0:
            return reps_N, N_weights, G2rVN, N2G, G2num_N, E, E_weights, reps_E

        ### Connecting Nodes and VNs

        vreps_cVN = torch.index_select(self.vreps_VN, 0, G2VN_cands[chosen_G2VN == 1, 1])

        nN_weights = torch.cat([N_weights, chosen_scores], dim = 0)

        nN2G = torch.cat([N2G, chosen_Gs], dim = 0)

        nG2num_VN = torch.zeros_like(G2num_N).index_add(dim = 0, index = chosen_Gs, source = chosen_scores)

        nG2num_N = G2num_N + nG2num_VN

        VN2cVN = -1 * torch.ones(num_VN_cands, device = device).long()
        VN2cVN[chosen_G2VN == 1] = torch.arange(num_cVN, device = device)

        #filtered E cands based on the chosen VNs
        fE_cands = E_cands[chosen_G2VN[E_cands[:,1]] == 1]
        fE_cands[:,1] = VN2cVN[fE_cands[:,1]] 
        fE_cands_init_logit_per_head = E_cands_init_logit_per_head[chosen_G2VN[E_cands[:,1]] == 1]

        logsoft_N = GroupLogSoftmax(num_groups = num_N, groups = fE_cands[:,0], logits = fE_cands_init_logit_per_head)
        logsoft_VN = GroupLogSoftmax(num_groups = num_cVN, groups = fE_cands[:,1], logits = fE_cands_init_logit_per_head)

        alpha = F.softmax(self.alpha, dim = 0).unsqueeze(dim = 1)

        fE_cands_logit_per_head = log_w * (alpha[0] * logsoft_N + alpha[1] * logsoft_VN) + fE_cands_init_logit_per_head

        fE_cands_logits = fE_cands_logit_per_head.mean(dim = -1)
        
        if self.training:
            fE_cands_probs = GumbelSigmoid(fE_cands_logits, tau = gumbel_temperature)
        else:
            fE_cands_probs = F.sigmoid(fE_cands_logits)
        
        chosen_E_cands_final = torch.where(fE_cands_probs >= threshold, 1, 0) - fE_cands_probs.detach() + fE_cands_probs

        chosen_E_scores = chosen_E_cands_final[chosen_E_cands_final == 1]
        chosen_src = fE_cands[chosen_E_cands_final == 1, 0]
        chosen_dst = fE_cands[chosen_E_cands_final == 1, 1]

        ## Section 4.2

        if "sigmoid" in self.aggr_VN:
            msg = torch.zeros_like(vreps_cVN).index_add(dim = 0, index = chosen_dst, source = F.sigmoid(fE_cands_logits[chosen_E_cands_final == 1]).unsqueeze(dim = 1) * torch.index_select(reps_N, 0, chosen_src))
            if self.aggr_VN == "normalized_sigmoid":
                sum_nbs = torch.zeros(num_cVN, 1, device = device).index_add(dim = 0, index = chosen_dst, source = F.sigmoid(fE_cands_logits[chosen_E_cands_final == 1]).unsqueeze(dim = 1))
                msg = msg / (sum_nbs + 1e-6)
            elif self.aggr_VN != "sigmoid":
                raise NotImplementedError
        elif self.aggr_VN in ["mean", "sum"]:
            msg = torch.zeros_like(vreps_cVN).index_add(dim = 0, index = fE_cands[:,1], source = chosen_E_cands_final.unsqueeze(dim = 1) * torch.index_select(reps_N, 0, fE_cands[:,0]))
            if self.aggr_VN == "mean":
                cnt_nbs = torch.zeros(num_cVN, 1, device = device).index_add(dim = 0, index = fE_cands[:,1], source = chosen_E_cands_final.unsqueeze(dim = 1))
                cnt_nbs = torch.where(cnt_nbs == 0, 1, cnt_nbs)
                msg = msg / cnt_nbs
        else:
            raise NotImplementedError
        
        beta = F.softmax(self.beta, dim = 0).unsqueeze(dim = 1)

        reps_cVN = beta[0] * vreps_cVN + beta[1] * msg
        
        nreps_N = torch.cat([reps_N, reps_cVN], dim = 0)

        nE = torch.stack([chosen_src, chosen_dst + num_N], dim = 1)
        nE = torch.cat([E, nE, torch.fliplr(nE)], dim = 0)

        nE_weights = torch.cat([E_weights, chosen_E_scores, chosen_E_scores], dim = 0)

        nreps_E = torch.cat([reps_E, nE_weights[num_E:].unsqueeze(dim = 1) * self.init_nE.repeat(2*chosen_E_cands_final.sum().long(), 1)])

        ## Section 4.3        

        VE_cands_src = torch.arange(num_cVN, device = device).repeat_interleave(nG2num_VN[chosen_Gs].long())

        VN_start_indices_per_G =  torch.cat([torch.zeros(1,device = device).long(), torch.cumsum(nG2num_VN.long(), dim = 0)[:-1]], dim = 0)
        VN_start_indices_per_N = VN_start_indices_per_G[chosen_Gs]
        start_index_per_VN = torch.cat([torch.zeros(1,device = device).long(), torch.cumsum(nG2num_VN[chosen_Gs].long(), dim = 0)[:-1]], dim = 0)
        reiter_per_VN = torch.arange((nG2num_VN * nG2num_VN).sum().long(), device = device) - start_index_per_VN.repeat_interleave(nG2num_VN[chosen_Gs].long())
        VE_cands_dst = reiter_per_VN + VN_start_indices_per_N.repeat_interleave(nG2num_VN[chosen_Gs].long())

        VN2VN_cands = torch.stack([VE_cands_src[VE_cands_src < VE_cands_dst], VE_cands_dst[VE_cands_src < VE_cands_dst]], dim = 1)

        if self.norm_N in ["LayerNorm", "BatchNorm"]:
            norm_reps_cVN = self.norm_N_ftn(nreps_N)[num_N:]
        elif self.norm_N in ["GraphNorm"]:
            norm_reps_cVN = (GroupNormalize(cnt_groups = nG2num_N, groups = nN2G, vectors = nreps_N, mean_weights = self.c) * self.a  + self.b)[num_N:]
        elif self.norm_N in ["None"]:
            norm_reps_cVN = reps_cVN
        else:
            raise NotImplementedError
        
        VE_cands_init_logit_per_head = (torch.index_select(self.Q(norm_reps_cVN), 0, VN2VN_cands[:,0]) * torch.index_select(self.Q(norm_reps_cVN), 0, VN2VN_cands[:,1])).view(-1, self.num_head, self.dim_per_head).sum(dim = -1) / math.sqrt(self.dim_per_head)

        VE_cands_logsoftmax_VNwise = GroupLogSoftmax(num_groups = num_cVN, groups = torch.cat([VN2VN_cands[:,0], VN2VN_cands[:,1]], dim = 0), logits = torch.cat([VE_cands_init_logit_per_head, VE_cands_init_logit_per_head], dim = 0))

        VE_cands_logit_src_per_head = VE_cands_logsoftmax_VNwise[:len(VN2VN_cands)]
        VE_cands_logit_dst_per_head = VE_cands_logsoftmax_VNwise[len(VN2VN_cands):]

        VE_cands_logit_per_head = log_w * (VE_cands_logit_src_per_head + VE_cands_logit_dst_per_head)/2 + VE_cands_init_logit_per_head
        VN2VN_cands_logits = VE_cands_logit_per_head.mean(dim = -1)

        if self.training:
            VN2VN_cands_probs = GumbelSigmoid(VN2VN_cands_logits, tau = gumbel_temperature)
        else:
            VN2VN_cands_probs = F.sigmoid(VN2VN_cands_logits)
        
        chosen_VN2VN_cands = torch.where(VN2VN_cands_probs >= threshold, 1, 0) - VN2VN_cands_probs.detach() + VN2VN_cands_probs
        chosen_VN2VN_scores = chosen_VN2VN_cands[chosen_VN2VN_cands == 1]
        chosen_VN2VN_src = VN2VN_cands[chosen_VN2VN_cands == 1, 0]
        chosen_VN2VN_dst = VN2VN_cands[chosen_VN2VN_cands == 1, 1]

        nVN2VN = torch.stack([chosen_VN2VN_src + num_N, chosen_VN2VN_dst + num_N], dim = 1)
        nE = torch.cat([nE, nVN2VN, torch.fliplr(nVN2VN)], dim = 0)
        nE_weights = torch.cat([nE_weights, chosen_VN2VN_scores, chosen_VN2VN_scores], dim = 0)
        nVN2VN_weights = chosen_VN2VN_scores.unsqueeze(dim = 1) * self.init_nVN2VN.repeat(chosen_VN2VN_cands.sum().long(), 1)
        nreps_E = torch.cat([nreps_E, nVN2VN_weights, nVN2VN_weights])

        print(f"avg VEs: {(chosen_E_cands_final.sum().long()+chosen_VN2VN_cands.sum().long())/num_G}")

        return nreps_N, nN_weights, G2rVN, nN2G, nG2num_N, nE, nE_weights, nreps_E

class MAVN(GNN):
    def __init__(self, dim_in = 235, dim_out = 235, dim_mlp = 235, dim_dot = 235, dim_pe = 28, task = "Graph Classification", num_pe = 20, 
                    num_class = 1, num_layer = 6, num_mlp_layer = 1, num_pred_mlp_layer = 3, num_head = 1, num_VN = 40,
                    act = "ReLU", dropout = 0.1, normalize = "None", base = "GCN", readout = "mean",
                    N_feat_type = "Categorical", num_N_feat = 10, N_feat_cats = [],
                    E_feat_type = "Categorical", num_E_feat = 3, E_feat_cats = [],
                    pe_type = "RWSE", pe_norm = "Batch", pred_mlp_layer_init = "xavier_normal", normalize_before_pred = True, res = True, norm_N = "None", norm_VN = "None", aggr_VN = "normalized_sigmoid", log_w = 1.0):
        super(MAVN, self).__init__(dim_in = dim_in, dim_out = dim_out, dim_mlp = dim_mlp, dim_pe = dim_pe, task = task, num_pe = num_pe,
                                             num_class = num_class, num_layer = num_layer, num_pred_mlp_layer = num_pred_mlp_layer,
                                             act = act, dropout = dropout, normalize = normalize, base = base, readout = readout,
                                             N_feat_type = N_feat_type, num_N_feat = num_N_feat, N_feat_cats = N_feat_cats,
                                             E_feat_type = E_feat_type, num_E_feat = num_E_feat, E_feat_cats = E_feat_cats,
                                             pe_type = pe_type, pe_norm = pe_norm, pred_mlp_layer_init = pred_mlp_layer_init, normalize_before_pred = normalize_before_pred, res = res)
        mpgenerator_list = []
        for l in range(self.num_layer):
            mpgenerator_list.append(MessagePathGenerator(dim = self.dim_in[l], dim_dot = dim_dot, dim_mlp = dim_mlp, num_mlp_layer = num_mlp_layer, num_head = num_head, dropout = dropout, act = act, \
                                                         norm_N = norm_N, norm_VN = norm_VN, aggr_VN = aggr_VN, num_VN = num_VN, log_w = log_w))
        self.MessagePathGenerators = nn.ModuleList(mpgenerator_list)
        self.num_VN = num_VN

        if task in ["Graph Classification", "Graph Regression"]:
            self.global_VN = nn.Parameter(torch.zeros(1, self.dim_in[0]))
            self.global_E = nn.Parameter(torch.zeros(1, self.dim_in[0]))
            nn.init.kaiming_uniform_(self.global_VN.unsqueeze(dim = 0), a = math.sqrt(5))
            nn.init.kaiming_uniform_(self.global_E.unsqueeze(dim = 0), a = math.sqrt(5))

    def forward(self, N2G: torch.LongTensor, E2G: torch.LongTensor, G2num_N: torch.LongTensor, \
                feat_N: torch.FloatTensor, feat_E:torch.FloatTensor, E: torch.LongTensor, pe_feat_N: torch.FloatTensor, num_G: int,
                gumbel_temperature: float = 1, threshold: float = 0.5):

        num_G = len(G2num_N)
        num_N = len(feat_N)
        num_E = len(E)

        if self.pe_type == "LapPE":
            if self.training:
                pe_feat_N[:,(torch.rand(self.num_pe) < 0.5),0] *= -1
                rep_pe = torch.sum(self.enc_pe(self.pe_norm(pe_feat_N)), dim = 1)
            else:
                rep_pe = torch.sum(self.enc_pe(self.pe_norm(pe_feat_N)), dim = 1)
        elif self.pe_type in ["None", "RWSE"]:
            rep_pe = self.enc_pe(self.pe_norm(pe_feat_N))
        elif self.pe_type == "LapPE+RWSE":
            lap = pe_feat_N[0]
            if self.training:
                lap[:,(torch.rand(self.num_pe[0]) < 0.5),0] *= -1
                rep_lap = torch.sum(self.enc_pe1(self.pe_norm1(lap)), dim = 1)
            else:
                rep_lap = torch.sum(self.enc_pe1(self.pe_norm1(lap)), dim = 1)
            rwse = pe_feat_N[1]
            rep_rwse = self.enc_pe2(self.pe_norm2(rwse))
            rep_pe = torch.cat([rep_lap, rep_rwse], dim = 1)
        else:
            raise NotImplementedError

        reps_N = torch.cat([self.enc_N(feat_N), rep_pe], dim = 1)
        N_weights = torch.ones((num_N,)).cuda()
        G2rVN = torch.ones((num_G, self.num_VN)).bool().cuda()
        G2num_N = G2num_N.float()
        E_weights = torch.ones((num_E,)).cuda()
        reps_E = self.enc_E(feat_E)

        if self.task in ["Graph Classification", "Graph Regression"]:
            reps_N = torch.cat([reps_N, self.global_VN.repeat(num_G, 1)], dim = 0)
            N_weights = torch.ones((num_N+num_G)).cuda()
            
            E_weights = torch.ones((num_E + 2*num_N,)).cuda()
            gE = torch.stack([torch.arange(num_N).cuda(), N2G + num_N], dim = 1)
            E = torch.cat([E, gE, torch.fliplr(gE)], dim = 0)
            reps_E = torch.cat([reps_E, self.global_E.repeat(2*num_N, 1)])
            N2G = torch.cat([N2G, torch.arange(num_G).cuda()])
            G2global = torch.arange(num_G).cuda() + num_N
            G2num_N = G2num_N + 1
            num_E = num_E + 2*num_N
            num_N = num_N + num_G

        VNs_per_layer = []
        VEs_per_layer = []
        VN_freqs_per_layer = []
        rem_VN_freqs = [G2rVN[:,idx].sum() for idx in range(self.num_VN)]


        for l, (layer, mpgen) in enumerate(zip(self.layers, self.MessagePathGenerators)):
            print(f"---layer #{l}---")            

            ## Section 4.1 - 4.3
            reps_N, N_weights, G2rVN, N2G, G2num_N, E, E_weights, reps_E = mpgen(gumbel_temperature, threshold,
                                                                                 reps_N, N_weights, G2rVN, N2G, G2num_N, E, E_weights, reps_E)

            VNs_per_layer.append(len(reps_N) - num_N - sum(VNs_per_layer))
            VEs_per_layer.append((len(reps_E) - num_E - 2*sum(VEs_per_layer))/2)
            VN_freqs_per_layer.append([(rem_VN_freqs[idx] - G2rVN[:,idx].sum()).long() for idx in range(self.num_VN)])
            rem_VN_freqs = [G2rVN[:,idx].sum() for idx in range(self.num_VN)]

            ## Section 4.4
            reps_N, reps_E = layer(reps_N = reps_N, node_weights = N_weights, edges = E, edge_weights = E_weights, reps_E = reps_E)

            if reps_E.shape[-1] != reps_N.shape[-1]:
                reps_E = torch.cat([reps_E, torch.zeros(reps_E.shape[0], reps_N.shape[1] - reps_E.shape[1]).cuda()], dim = 1)
                                                                            
        VNs_per_layer = torch.tensor(VNs_per_layer).cuda()
        VEs_per_layer = torch.tensor(VEs_per_layer).cuda()
        VN_freqs_per_layer = torch.tensor(VN_freqs_per_layer).cuda()
        if self.task in ["Graph Classification", "Graph Regression"]:
            pred = self.mlp(self.normalize(torch.index_select(reps_N, 0, G2global)))
        elif self.task in ["Node Classification"]:
            pred = self.mlp(self.normalize(reps_N[:num_N]))
        print(f"#VNs:{sum(VNs_per_layer).item()} #VEs: {sum(VEs_per_layer).item()} VN freqs: {VN_freqs_per_layer.sum(dim = 0).detach().cpu().tolist()}")
        return pred, VNs_per_layer, VEs_per_layer, VN_freqs_per_layer