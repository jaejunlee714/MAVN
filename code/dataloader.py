import torch
from torch.utils.data import Dataset
import numpy as np
import os
import logging
import pickle

from typing import List, Optional, Union, Annotated, Dict, Any
from tqdm import tqdm

from positional_encoding import RWSE, LapPE
from constants import NODE_FEAT_TYPE_DICT, EDGE_FEAT_TYPE_DICT

class Custom_Dataset_Multi(Dataset):
    def __init__(self, dataset_dir: str, dataset_name: str, logger: logging.Logger, pe: Annotated[str, "RWSE", "LapPE", "None"] = "RWSE", num_pe: int = 20):
        '''
        Dataset consisting of multiple graphs
        '''
        self.dataset_dir = os.path.join(dataset_dir, dataset_name)
        self.dataset_name = dataset_name
        self.logger = logger
        self.pe = pe
        self.num_pe = num_pe

        self.num_N_train, self.num_E_train, self.G_train, self.num_class, self.N_feat_num, self.E_feat_num, self.N_feat_cats, self.E_feat_cats = self.parse(split = "train")
        self.num_N_valid, self.num_E_valid, self.G_valid = self.parse(split = "valid")
        self.num_N_test, self.num_E_test, self.G_test = self.parse(split = "test")
        
        self.N_feat_type = NODE_FEAT_TYPE_DICT[dataset_name]
        self.E_feat_type = EDGE_FEAT_TYPE_DICT[dataset_name]

        self.logger.info("Dataset Loaded!")
        self.num_train = len(self.G_train)
        self.num_valid = len(self.G_valid)
        self.num_test = len(self.G_test)
        total_graphs = self.num_train + self.num_valid + self.num_test
        total_nodes = self.num_N_train + self.num_N_valid + self.num_N_test
        total_edges = self.num_E_train + self.num_E_valid + self.num_E_test
        self.logger.info(f"total graphs:{total_graphs}, total nodes:{total_nodes}, avg nodes:{total_nodes/total_graphs}, "+
                         f"mean deg:{total_edges/total_nodes}, total edges:{total_edges}, avg edges:{total_edges/total_graphs}")
        self.logger.info(f"train graphs:{self.num_train}, train nodes:{self.num_N_train}, avg nodes:{self.num_N_train/self.num_train}, "+
                         f"mean deg:{self.num_E_train/self.num_N_train}, total edges:{self.num_E_train}, avg edges:{self.num_E_train/self.num_train}")
        self.logger.info(f"valid graphs:{self.num_valid}, valid nodes:{self.num_N_valid}, avg nodes:{self.num_N_valid/self.num_valid}, "+
                         f"mean deg:{self.num_E_valid/self.num_N_valid}, total edges:{self.num_E_train}, avg edges:{self.num_E_valid/self.num_valid}")
        self.logger.info(f"test graphs:{self.num_test}, test nodes:{self.num_N_test}, avg nodes:{self.num_N_test/self.num_test}, "+
                         f"mean deg:{self.num_E_test/self.num_N_test}, total edges:{self.num_E_test}, avg edges:{self.num_E_test/self.num_test}")
    def parse(self, split = "train"):
        raise NotImplementedError

    def get_train(self, idxs):
        idxs = sorted(idxs)
        graphs = [self.G_train[idx] for idx in idxs]
        return self.get_batch(idxs, graphs, self.num_N_train)

    def get_valid(self, idxs):
        graphs = [self.G_valid[idx] for idx in idxs]
        return self.get_batch(idxs, graphs, self.num_N_valid)

    def get_test(self, idxs):
        graphs = [self.G_test[idx] for idx in idxs]
        return self.get_batch(idxs, graphs, self.num_N_test)

    def get_batch(self, idxs, graphs, num_N):
        num_Ns = torch.tensor([len(graph[0]) for graph in graphs]).long()
        Ns = torch.cat([graph[0] for graph in graphs])
        N2idx = -1 * torch.ones(num_N).long()
        N2idx[Ns] = torch.arange(torch.sum(num_Ns))
        N2G = torch.arange(len(idxs)).repeat_interleave(num_Ns)
        Es = torch.cat([N2idx[graph[1]] for graph in graphs])
        assert (-1 == Es).sum() == 0
        assert (N2G[Es][:,0] != N2G[Es][:,1]).sum() == 0
        E2G = N2G[Es][:,0]

        N_feats = torch.cat([graph[2] for graph in graphs])
        E_feats = torch.cat([graph[3] for graph in graphs])
        if self.pe != "LapPE+RWSE":
            pes = torch.cat([graph[4] for graph in graphs])
        else:
            lap = torch.cat([graph[4][0] for graph in graphs])
            rwse = torch.cat([graph[4][1] for graph in graphs])
            pes = [lap, rwse]
        ys = torch.cat([graph[5] for graph in graphs])
        
        if "+" in self.pe:
            pes_cuda = [pes[0].cuda(), pes[1].cuda()]
        else:
            pes_cuda = pes.cuda()
        return N2G.cuda(), E2G.cuda(), num_Ns.cuda(), N_feats.cuda(), E_feats.cuda(), Es.cuda(), pes_cuda, ys.cuda()
        

# Peptides-func, Peptides-struct
class LRGB_GraphTask_Dataset(Custom_Dataset_Multi):
    def __init__(self, dataset_dir: str, dataset_name: str, logger: logging.Logger, pe: Annotated[str, "RWSE", "None"] = "RWSE", num_pe: int = 20):
        super(LRGB_GraphTask_Dataset, self).__init__(dataset_dir = dataset_dir, dataset_name = dataset_name, logger = logger, pe = pe, num_pe = num_pe)

    def parse(self, split = "train"):
        self.logger.info(f"Loading {split} set")
        graphs = torch.load(f"{self.dataset_dir}/{split}.pt")
        if self.pe == "RWSE":
            if os.path.exists(f"{self.dataset_dir}/{split}_RWSE_{self.num_pe}.pt"):
                pes = torch.load(f"{self.dataset_dir}/{split}_RWSE_{self.num_pe}.pt")
            else:
                pes = []
                for graph in tqdm(graphs):
                    N_feat = graph[0]
                    E = graph[2].T
                    pes.append(RWSE(len(N_feat), E, self.num_pe))
                torch.save(pes, f"{self.dataset_dir}/{split}_RWSE_{self.num_pe}.pt")
        elif self.pe == "LapPE":
            if os.path.exists(f"{self.dataset_dir}/{split}_LapPE_{self.num_pe}.pt"):
                pes = torch.load(f"{self.dataset_dir}/{split}_LapPE_{self.num_pe}.pt")
            else:
                pes = []
                for graph in tqdm(graphs):
                    N_feat = graph[0]
                    E = graph[2].T
                    pes.append(LapPE(len(N_feat), E, self.num_pe))
                torch.save(pes, f"{self.dataset_dir}/{split}_LapPE_{self.num_pe}.pt")
        elif self.pe == "LapPE+RWSE":
            pes = []
            if os.path.exists(f"{self.dataset_dir}/{split}_LapPE_{self.num_pe[0]}.pt"):
                pes.append(torch.load(f"{self.dataset_dir}/{split}_LapPE_{self.num_pe[0]}.pt"))
            else:
                pe = []
                for graph in tqdm(graphs):
                    N_feat = graph[0]
                    E = graph[2].T
                    pe.append(LapPE(len(N_feat), E, self.num_pe[0]))
                torch.save(pe, f"{self.dataset_dir}/{split}_LapPE_{self.num_pe[0]}.pt")
                pes.append(pe)

            if os.path.exists(f"{self.dataset_dir}/{split}_RWSE_{self.num_pe[1]}.pt"):
                pes.append(torch.load(f"{self.dataset_dir}/{split}_RWSE_{self.num_pe[1]}.pt"))
            else:
                pe = []
                for graph in tqdm(graphs):
                    N_feat = graph[0]
                    E = graph[2].T
                    pe.append(RWSE(len(N_feat), E, self.num_pe[1]))
                torch.save(pe, f"{self.dataset_dir}/{split}_RWSE_{self.num_pe[1]}.pt")
                pes.append(pe)
        elif self.pe == "None":
            pes = [torch.empty((len(graph[0]), 0)) for graph in graphs]
        else:
            raise NotImplementedError
        num_N = 0
        num_E = 0
        G = []
        if split == "train":
            num_class = -1
            N_feat_num = len(graphs[0][0][0])
            E_feat_num = len(graphs[0][1][0])
            N_feat_cats = torch.zeros(N_feat_num).long()
            E_feat_cats = torch.zeros(E_feat_num).long()
        for idx, graph in enumerate(tqdm(graphs)):
            N_feat = graph[0]
            E_feat = graph[1]
            if split == "train":
                N_feat_cats = torch.maximum(N_feat_cats, torch.max(N_feat, dim = 0)[0])
                E_feat_cats = torch.maximum(E_feat_cats, torch.max(E_feat, dim = 0)[0])
            E = graph[2].T
            num_E += len(E)
            E = E + num_N
            N = torch.arange(num_N, num_N + len(graph[0]))
            num_N = num_N + len(graph[0])
            y = graph[3]
            if split == "train" and num_class == -1:
                num_class = y.size()[-1]
            elif split != "train":
                assert y.size()[-1] == self.num_class
            else:
                assert y.size()[-1] == num_class
            if self.pe == "LapPE+RWSE":
                G.append((N, E, N_feat, E_feat, [pes[0][idx], pes[1][idx]], y))
            else:
                G.append((N, E, N_feat, E_feat, pes[idx], y))
        if split == "train":
            N_feat_cats = N_feat_cats + 1
            E_feat_cats = E_feat_cats + 1
            return num_N, num_E, G, num_class, N_feat_num, E_feat_num, N_feat_cats, E_feat_cats
        else:
            return num_N, num_E, G

# PascalVOC-SP, COCO-SP
class LRGB_NodeTask_Dataset(Custom_Dataset_Multi):
    def __init__(self, dataset_dir: str, dataset_name: str, logger: logging.Logger, pe: Annotated[str, "RWSE", "None"] = "RWSE", num_pe: int = 20):
        ## coco remap from Pytorch-Geometric
        if dataset_name == 'COCO-SP':
            self.orig_label_idx = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78,
            79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
            ]
            self.label_map = {}
            for idx, item in enumerate(self.orig_label_idx):
                self.label_map[item] = idx
            self.N_feat_mean = torch.tensor([
                                        4.6977347e-01, 4.4679317e-01, 4.0790915e-01, 7.0808627e-02,
                                        6.8686441e-02, 6.8498217e-02, 6.7777938e-01, 6.5244222e-01,
                                        6.2096798e-01, 2.7554795e-01, 2.5910738e-01, 2.2901227e-01,
                                        2.4261935e+02, 2.8985367e+02
                                        ]).unsqueeze(dim = 0)
            self.N_feat_std = torch.tensor([
                                       2.6218116e-01, 2.5831082e-01, 2.7416739e-01, 5.7440419e-02,
                                       5.6832556e-02, 5.7100497e-02, 2.5929087e-01, 2.6201612e-01,
                                       2.7675411e-01, 2.5456995e-01, 2.5140920e-01, 2.6182330e-01,
                                       1.5152475e+02, 1.7630779e+02
                                       ]).unsqueeze(dim = 0)
            self.E_feat_mean = torch.tensor([0.07848548, 43.68736]).unsqueeze(dim = 0)
            self.E_feat_std = torch.tensor([0.08902349, 28.473562]).unsqueeze(dim = 0)

        elif dataset_name == 'PascalVOC-SP':
            self.N_feat_mean = torch.tensor([
                                        4.5824501e-01, 4.3857411e-01, 4.0561178e-01, 6.7938097e-02,
                                        6.5604292e-02, 6.5742709e-02, 6.5212941e-01, 6.2894762e-01,
                                        6.0173863e-01, 2.7769071e-01, 2.6425251e-01, 2.3729359e-01,
                                        1.9344997e+02, 2.3472206e+02
                                        ]).unsqueeze(dim = 0)
            self.N_feat_std = torch.tensor([
                                       2.5952947e-01, 2.5716761e-01, 2.7130592e-01, 5.4822665e-02,
                                       5.4429270e-02, 5.4474957e-02, 2.6238337e-01, 2.6600540e-01,
                                       2.7750680e-01, 2.5197381e-01, 2.4986187e-01, 2.6069802e-01,
                                       1.1768297e+02, 1.4007195e+02
                                       ]).unsqueeze(dim = 0)
            self.E_feat_mean = torch.tensor([0.07640745, 33.73478]).unsqueeze(dim = 0)
            self.E_feat_std = torch.tensor([0.0868775, 20.945076]).unsqueeze(dim = 0)

        else:
            raise NotImplementedError

        super(LRGB_NodeTask_Dataset, self).__init__(dataset_dir = dataset_dir, dataset_name = dataset_name, logger = logger, pe = pe, num_pe = num_pe)

    def parse(self, split = "train"):

        self.logger.info(f"Loading {split} set")
        with open(f"{self.dataset_dir}/{split}.pickle" , "rb") as f:
            graphs = pickle.load(f)

        if self.pe == "RWSE":
            if os.path.exists(f"{self.dataset_dir}/{split}_RWSE_{self.num_pe}.pt"):
                pes = torch.load(f"{self.dataset_dir}/{split}_RWSE_{self.num_pe}.pt")
            else:
                pes = []
                for graph in tqdm(graphs):
                    N_feat = graph[0]
                    E = graph[2].T
                    pes.append(RWSE(len(N_feat), E, self.num_pe))
                torch.save(pes, f"{self.dataset_dir}/{split}_RWSE_{self.num_pe}.pt")
        elif self.pe == "LapPE":
            if os.path.exists(f"{self.dataset_dir}/{split}_LapPE_{self.num_pe}.pt"):
                pes = torch.load(f"{self.dataset_dir}/{split}_LapPE_{self.num_pe}.pt")
            else:
                pes = []
                for graph in tqdm(graphs):
                    N_feat = graph[0]
                    E = graph[2].T
                    pes.append(LapPE(len(N_feat), E, self.num_pe))
                torch.save(pes, f"{self.dataset_dir}/{split}_LapPE_{self.num_pe}.pt")
        elif self.pe == "None":
            pes = [torch.empty((len(graph[0]), 0)) for graph in graphs]
        else:
            raise NotImplementedError

        num_N = 0
        num_E = 0
        G = []
        if split == "train":
            num_class = -1
            N_feat_num = len(graphs[0][0][0])
            E_feat_num = len(graphs[0][1][0])
            N_feat_cats = 0
            E_feat_cats = 0
        for idx, graph in enumerate(tqdm(graphs)):
            N_feat = (graph[0].to(torch.float) - self.N_feat_mean)/self.N_feat_std
            E_feat = (graph[1].to(torch.float) - self.E_feat_mean)/self.E_feat_std
            E = graph[2].T
            num_E += len(E)
            E = E + num_N
            N = torch.arange(num_N, num_N + len(graph[0]))
            num_N = num_N + len(graph[0])
            y = torch.LongTensor(graph[3])
            if self.dataset_name == "COCO-SP":
                y = torch.LongTensor([self.label_map[label.item()] for label in y])
            if split == "train":
                num_class = max(torch.max(y).item(), num_class)
            assert len(y) == len(N)
            G.append((N, E, N_feat, E_feat, pes[idx], y))
        if split == "train":
            num_class = num_class + 1
            return num_N, num_E, G, num_class, N_feat_num, E_feat_num, N_feat_cats, E_feat_cats
        else:
            return num_N, num_E, G

class Custom_Dataset_Single(Dataset):
    
    def __init__(self, dataset_dir: str, dataset_name: str, logger: logging.Logger, pe: Annotated[str, "RWSE", "LapPE", "None"] = "RWSE", num_pe: int = 20):
        '''
        Dataset consisting of a single graph (Node-level task only)
        '''
        self.dataset_dir = os.path.join(dataset_dir, dataset_name)
        self.dataset_name = dataset_name
        self.logger = logger
        self.pe = pe
        self.num_pe = num_pe
        
        # keys: num_N, num_E, N_feat, E_feat, E, N_train, N_valid, N_test, num_class
        data = self.parse()
        self.num_N = data["num_N"]
        self.num_E = data["num_E"]
        self.N_feat = data["N_feat"]
        self.E_feat = data["E_feat"]
        self.E = data["E"]
        self.N_train = data["N_train"]
        self.N_valid = data["N_valid"]
        self.N_test = data["N_test"]
        self.num_class = data["num_class"]
        self.y = data["y"]
        self.pes = data["pes"]
        
        self.N_feat_type = NODE_FEAT_TYPE_DICT[dataset_name]
        self.E_feat_type = EDGE_FEAT_TYPE_DICT[dataset_name]
        self.N_feat_num = self.N_feat.shape[-1]
        self.E_feat_num = self.E_feat.shape[-1]
        self.N_feat_cats = None # !
        self.E_feat_cats = None # !
        self.logger.info("Dataset Loaded!")
        
        self.num_train = len(self.N_train)
        self.num_valid = len(self.N_valid)
        self.num_test = len(self.N_test)
        
        assert (self.num_train == self.N_train.shape[0])
        assert (self.num_valid == self.N_valid.shape[0])
        assert (self.num_test == self.N_test.shape[0])
        
        total_graphs = 1
        total_nodes = self.num_train + self.num_valid + self.num_test
        total_edges = self.num_E
        self.logger.info(f"total graphs:{total_graphs}, total nodes:{total_nodes}, avg nodes:{total_nodes/total_graphs}, "+
                         f"mean deg:{total_edges/total_nodes}, total edges:{total_edges}, avg edges:{total_edges/total_graphs}")
    
    def parse(self) -> Dict[str, Any]:
        raise NotImplementedError
    
    def get_train(self, idxs):
        return self.get_batch(self.N_train[idxs])

    def get_valid(self, idxs):
        return self.get_batch(self.N_valid[idxs])

    def get_test(self, idxs):
        return self.get_batch(self.N_test[idxs])

    def get_batch(self, idxs):
        num_Ns = torch.tensor([self.num_N]).long()
        N2G = torch.zeros(self.num_N).long()
        Es = torch.tensor(self.E)
        E2G = torch.zeros(len(Es))

        N_feats = torch.tensor(self.N_feat)
        E_feats = torch.tensor(self.E_feat)
        pes = self.pes.clone().detach()
        ys = torch.tensor(self.y[idxs])
        
        return N2G.cuda(), E2G.cuda(), num_Ns.cuda(), N_feats.cuda(), E_feats.cuda(), Es.cuda(), pes.cuda(), ys.cuda()

class Heterophily_Dataset(Custom_Dataset_Single):
    
    def __init__(self, dataset_dir: str, dataset_name: str, logger: logging.Logger,
                 pe: Annotated[str, "RWSE", "None"] = "RWSE", num_pe: int = 20, split_idx: int = 0) -> None:
        '''
        Dataset from "A critical look at the evaluation of GNNs under heterophily: Are we really making progress? (ICLR 2023)"
        '''
        assert (split_idx < 10) and (split_idx >= 0), "split_idx for Heterophily_Dataset must be between 0 and 9"
        self.save_path = os.path.join(dataset_dir, dataset_name, f"{dataset_name}.npz")
        self.split_idx = split_idx
        super().__init__(dataset_dir=dataset_dir, dataset_name=dataset_name, logger=logger, pe=pe, num_pe=num_pe)
    
    def parse(self) -> Dict[str, Any]:
        # Load npz file containing the dataset
        data = np.load(self.save_path) # keys: 'node_features', 'node_labels', 'edges', 'train_masks', 'val_masks', 'test_masks'
        
        # To bidirectional graph
        edges = data["edges"]
        edges = np.concatenate([edges, edges[:,::-1]], axis=0)  
        edges = np.unique(edges, axis=0)
        
        # Select split based on self.split_idx
        train_mask = data["train_masks"][self.split_idx]
        val_mask = data["val_masks"][self.split_idx]
        test_mask = data["test_masks"][self.split_idx]
        
        y = data["node_labels"]

        # Create data dictionary
        data_dict = {}
        data_dict["num_N"] = data["node_features"].shape[0]
        data_dict["num_E"] = edges.shape[0]
        data_dict["N_feat"] = data["node_features"]
        data_dict["E_feat"] = np.zeros((edges.shape[0], 1), dtype=np.float32)
        data_dict["E"] = edges
        data_dict["N_train"] = train_mask.nonzero()[0]
        data_dict["N_valid"] = val_mask.nonzero()[0]
        data_dict["N_test"] = test_mask.nonzero()[0]
        data_dict["y"] = y
        data_dict["num_class"] = y.max() + 1
        
        # Positional Encodings
        pe_save_path = os.path.join(self.dataset_dir, f"{self.dataset_name}_{self.pe}_{self.num_pe}.pt")
        if os.path.exists(pe_save_path):
            pes = torch.load(pe_save_path)
        else:
            print(f"Positional Encodings not found at {pe_save_path}. Generating...")
            if self.pe == "RWSE":
                pes = RWSE(data_dict["num_N"], edges, self.num_pe)
            elif self.pe == "LapPE":
                pes = LapPE(data_dict["num_N"], edges, self.num_pe)
            elif self.pe == "None":
                pes = torch.empty((data_dict["num_N"], 0))
            else:
                raise NotImplementedError
            torch.save(pes, pe_save_path)
            print("Positional Encodings generated and saved!")
        data_dict["pes"] = pes
        
        return data_dict