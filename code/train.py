import os
import argparse
import logging
import datetime
import math
import random
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from typing import Optional, Tuple, Callable, Dict, Annotated, Union

from dataloader import Custom_Dataset_Multi, LRGB_GraphTask_Dataset, LRGB_NodeTask_Dataset, Heterophily_Dataset, Custom_Dataset_Single
from metrics import AP, Macro_F1, MAE, AUCROC, Accuracy
from constants import *
from models import GNN, MAVN

os.environ['OMP_NUM_THREADS']='8'
torch.set_num_threads(8)
torch.cuda.empty_cache()


class ExperimentManager:
    
    def __init__(self, args: argparse.Namespace,
                 logger: logging.Logger,
                 file_format: str) -> None:
        self.args = args
        self.logger = logger
        self.file_format = file_format
        self.data: Optional[Union[Custom_Dataset_Multi, Custom_Dataset_Single]] = None
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.task_loss_func: Optional[Callable] = None
    
    def set_seeds(self) -> None:
        '''
        Set seeds for reproducibility
        '''
        torch.manual_seed(self.args.seed)
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
    
    def load_dataset(self) -> None:
        '''
        Load dataset based on args
        '''
        args = self.args
        logger = self.logger
        
        # Check if dataset is available
        assert args.dataset_name in AVAILABLE_DATASETS, f"{args.dataset_name} is not available. List of available datasets: {AVAILABLE_DATASETS}"
        
        # Check if task is available for the dataset
        assert args.task in DATASET_TASK_DICT[args.dataset_name], f"{args.task} is not available for {args.dataset_name}. List of available tasks: {DATASET_TASK_DICT[args.dataset_name]}"
        
        if args.dataset_name in LRGB_GRAPH_DATASETS:
            data = LRGB_GraphTask_Dataset(args.dataset_dir, args.dataset_name, logger, pe = args.pe, num_pe = args.num_pe)
        elif args.dataset_name in LRGB_NODE_DATASETS:
            data = LRGB_NodeTask_Dataset(args.dataset_dir, args.dataset_name, logger, pe = args.pe, num_pe = args.num_pe)
        elif args.dataset_name in HETEROPHILY_DATASETS:
            data = Heterophily_Dataset(args.dataset_dir, args.dataset_name, logger, pe = args.pe, num_pe = args.num_pe, split_idx = args.split)
        else:
            raise NotImplementedError
        
        assert data is not None, "Dataset is not properly loaded"
        self.data = data
    
    def initialize_model(self) -> None:
        '''
        Initialize model based on args
        '''
        data = self.data
        args = self.args
        
        if args.model_name == "GNN":
            My_Model = GNN
        elif args.model_name == "MAVN":
            My_Model = MAVN
        else:
            raise NotImplementedError
        
        model = My_Model(
                dim_in = args.dim_in,
                dim_out = args.dim_out,
                dim_dot = args.dim_dot if args.dim_dot != -1 else args.dim_in,
                dim_pe = args.dim_pe,
                task = args.task,
                num_pe = args.num_pe,
                num_class = data.num_class,
                num_layer = args.num_layer,
                num_mlp_layer = args.num_mlp_layer,
                num_pred_mlp_layer = args.num_pred_mlp_layer,
                num_VN = args.num_VN,
                num_head = args.num_head,
                act = args.act,
                dropout = args.dropout,
                normalize = args.normalize,
                base = args.base_model,
                N_feat_type = data.N_feat_type, num_N_feat = data.N_feat_num, N_feat_cats = data.N_feat_cats,
                E_feat_type = data.E_feat_type, num_E_feat = data.E_feat_num, E_feat_cats = data.E_feat_cats,
                pe_type = args.pe,
                pe_norm = args.pe_norm,
                readout = args.readout,
                pred_mlp_layer_init = args.pred_mlp_layer_init,
                normalize_before_pred = args.normalize_before_pred,
                res = args.res,
                dim_mlp = args.dim_mlp,
                norm_N = args.norm_N,
                norm_VN = args.norm_VN,
                aggr_VN = args.aggr_VN,
                log_w = args.log_w
                ).cuda()
        
        assert model is not None, "Model is not properly initialized"
        self.model = model
    
    def setup(self) -> None:
        
        self.set_seeds()
        self.load_dataset()
        self.initialize_model()

        args = self.args
        model = self.model
        
        # Determine optimizer
        if args.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr_max, weight_decay = args.weight_decay)
        elif args.optimizer == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr = args.lr_max, weight_decay = args.weight_decay)
        else:
            raise NotImplementedError(f"{args.optimizer} is not implemented. Choose from {AVAILABLE_OPTIMIZERS}")
        self.optimizer = optimizer
        
        # Determine learning rate scheduler
        if args.scheduler == "CosLRLinearWarmupRestart":
            LinearWarmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: max(1e-6, epoch/(args.warmup_epoch-1)))
            CosLRRestart = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.restart_epoch, T_mult = args.restart_mult, eta_min = args.lr_min)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers = [LinearWarmup, CosLRRestart], milestones = [args.warmup_epoch])
        elif args.scheduler == "None":
            scheduler = None
        else:
            raise NotImplementedError(f"{args.scheduler} is not implemented. Choose from {AVAILABLE_SCHEDULERS}")
        self.scheduler = scheduler
        
        # Determine task loss
        if args.loss_function == "BCE":
            loss_function = F.binary_cross_entropy_with_logits
        elif args.loss_function == "CE":
            loss_function = F.cross_entropy
        elif args.loss_function == "MAE":
            loss_function = F.l1_loss
        else:
            raise NotImplementedError(f"{args.loss_function} is not implemented. Choose from {AVAILABLE_LOSSES}")
        self.task_loss_func = loss_function
        
        # Log args
        logger = self.logger
        for arg_name in vars(args).keys():
            logger.info(f"{arg_name}:{vars(args)[arg_name]}")
        logger.info("Args Listed!")
        logger.info(f"Num params:{sum(p.numel() for p in model.parameters())}")
        
        # Load checkpoint
        if args.start_epoch != 0:
            ckpt_path = f"./ckpt/{args.exp}/{args.dataset_name}/{self.file_format}_{args.start_epoch}.ckpt"
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            logging.info(f"Checkpoint loaded from {ckpt_path}")
        logging.info(model)
        optimizer.zero_grad()
    
    def get_tau(self, epoch: int) -> float:
        tau_start = self.args.tau_start
        tau_end = self.args.tau_end
        tau_epoch = self.args.tau_epoch
        args = self.args
        
        if epoch < args.warmup_epoch:
            tau = tau_start
        elif epoch < tau_epoch + args.warmup_epoch:
            tau = math.exp(math.log(tau_start) - (math.log(tau_start)-math.log(tau_end))/tau_epoch * (epoch - args.warmup_epoch))
        else:
            tau = self.args.tau_end
        return tau
    
    def train_epoch(self, epoch: int) -> None:
        
        args = self.args
        data = self.data
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        model.train()
        
        ############### For logging ###############
        epoch_task_loss = 0
        epoch_VNs = 0
        epoch_VEs = 0
        epoch_VNs_per_layer = torch.zeros(args.num_layer).cuda()
        epoch_VEs_per_layer = torch.zeros(args.num_layer).cuda()
        epoch_VN_freqs_per_layer = torch.zeros(args.num_layer, args.num_VN).cuda()
        ###########################################
        

        tau = self.get_tau(epoch)
        batch_size = args.batch_size if args.batch_size != -1 else data.num_train
        for rand_idxs in tqdm(torch.split(torch.randperm(data.num_train), batch_size)):
            N2G, E2G, G2num_N, N_feat, E_feat, E, pe, y = data.get_train(rand_idxs)
            num_G = 1 if data.dataset_name in SINGLE_GRAPH_DATASETS else len(rand_idxs)
            
            pred, VNs_per_layer, VEs_per_layer, VN_freqs_per_layer = model(
                N2G, E2G, G2num_N, N_feat, E_feat, E, pe, num_G, gumbel_temperature = tau, threshold = args.threshold)
            
            if self.data.dataset_name in SINGLE_GRAPH_DATASETS:
                n_ids = self.data.N_train[rand_idxs]
                pred = pred[n_ids]
                
            loss, loss_dict = self.loss_function(pred, y)
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            
            optimizer.step()
            optimizer.zero_grad()
            
            ############### For logging ###############
            task_loss = loss_dict['task_loss']
            epoch_task_loss += task_loss
            
            epoch_VNs += (VNs_per_layer.sum()).item()
            epoch_VEs += (VEs_per_layer.sum()).item()
            epoch_VNs_per_layer += VNs_per_layer
            epoch_VEs_per_layer += VEs_per_layer
            epoch_VN_freqs_per_layer += VN_freqs_per_layer
            ###########################################

        if scheduler is not None:
            scheduler.step()
        # Logging results of train epoch
        num_graphs = 1 if data.dataset_name in SINGLE_GRAPH_DATASETS else data.num_train 
        logger.info(f"Epoch {epoch+1} GPU:{torch.cuda.max_memory_allocated()} Task Loss:{epoch_task_loss:.3f} #avg_VN:{epoch_VNs/num_graphs:.3f} #avg_VE:{epoch_VEs/num_graphs}")
        for l in range(args.num_layer):
            logger.info(f"Epoch {epoch+1} Layer {l} (Train) VNs:{epoch_VNs_per_layer[l].item()/num_graphs}, VEs:{epoch_VEs_per_layer[l].item()/num_graphs}, VN freqs:{(epoch_VN_freqs_per_layer[l].detach().cpu()/num_graphs).tolist()}")
        logger.info(f"Epoch {epoch+1} (Train) VN freqs:{(epoch_VN_freqs_per_layer.sum(dim = 0).detach().cpu()/num_graphs).tolist()}")

    
    def validation(self, epoch: int) -> Dict[str, float]:
        results = self.check_performance(epoch, flag = "Valid")
        return results
    
    def test(self, epoch: int) -> Dict[str, float]:
        results = self.check_performance(epoch, flag = "Test")
        return results
    
    def loss_function(self, pred: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        args = self.args
        data = self.data
        model = self.model
        
        # task loss
        if args.weighted_loss:
            assert "Classification" in args.task
            class_cnt = torch.zeros(data.num_class).cuda().index_add(dim = 0, index = y, source = torch.ones(len(y)).cuda())
            weights = (len(y) - class_cnt).float() / len(y) * (class_cnt != 0)
            task_loss = self.task_loss_func(pred, y, weight = weights)
        else:
            if pred.shape[1] == 2 and len(y.shape) == 1: # questions
                y = F.one_hot(y, num_classes = 2).float()
            task_loss = self.task_loss_func(pred, y)
        loss = task_loss
        
        loss_dict = {
            "task_loss": task_loss.item()
        }
        
        return loss, loss_dict
    
    def check_performance(self, epoch: int, flag: Annotated[str, "Valid", "Test"] = "Valid") -> Dict[str, float]:
        
        assert flag in ["Valid", "Test"], "flag must be either Valid or Test"
        
        args = self.args
        logger = self.logger        
        data = self.data
        model = self.model
        
        model.eval()
        with torch.no_grad():
            if args.multi_label:
                preds = np.empty((0, data.num_class))
                ys = np.empty((0, data.num_class))
            else:
                preds = np.empty((0))
                ys = np.empty((0))
                
            ################ For logging ################
            epoch_VNs_per_layer = torch.zeros(args.num_layer).cuda()
            epoch_VEs_per_layer = torch.zeros(args.num_layer).cuda()
            epoch_VN_freqs_per_layer = torch.zeros(args.num_layer, args.num_VN).cuda()
            num_vns = 0
            num_ves = 0
            #############################################
            
            # Get predictions
            num_graphs = data.num_valid if flag == "Valid" else data.num_test
            
            if data.dataset_name in SINGLE_GRAPH_DATASETS:
                num_graphs = 1
                idxs = torch.arange(data.num_valid) if flag == "Valid" else torch.arange(data.num_test)
                n_ids = data.N_valid[idxs] if flag == "Valid" else data.N_test[idxs]
                
                N2G, E2G, G2num_N, N_feat, E_feat, E, pe, y = data.get_valid(idxs) if flag == "Valid" else data.get_test(idxs)
                
                pred, VNs_per_layer, VEs_per_layer, VN_freqs_per_layer = model(N2G, E2G, G2num_N, N_feat, E_feat, E, pe, len(idxs), threshold = args.threshold)
                
                # Classification Only
                assert "Node Classification" in args.task, "Only node classification task is supported for Single Graph Datasets"
                if args.eval_metric in ["AUCROC"]:
                    pred = F.softmax(pred, dim = 1)[n_ids, 1]
                    y = y.squeeze(-1)
                elif args.eval_metric in ["Accuracy"]:
                    pred = torch.argmax(pred, dim = 1)[n_ids]
                else:
                    raise NotImplementedError
                
                preds = np.concatenate((preds, pred.detach().cpu().numpy()), axis = 0)
                ys = np.concatenate((ys, y.detach().cpu().numpy()), axis = 0)
                
                epoch_VNs_per_layer += VNs_per_layer
                epoch_VEs_per_layer += VEs_per_layer
                epoch_VN_freqs_per_layer += VN_freqs_per_layer
                num_vns += VNs_per_layer.sum().item()
                num_ves += VEs_per_layer.sum().item()
                
            else:
                for idxs in tqdm(torch.split(torch.arange(num_graphs), args.val_size)):
                    
                    N2G, E2G, G2num_N, N_feat, E_feat, E, pe, y = data.get_valid(idxs) if flag == "Valid" else data.get_test(idxs)
                    
                    pred, VNs_per_layer, VEs_per_layer, VN_freqs_per_layer = model(N2G, E2G, G2num_N, N_feat, E_feat, E, pe, len(idxs), threshold = args.threshold)
                    if args.multi_label and "Classification" in args.task:
                        pred = torch.sigmoid(pred)
                        if args.eval_metric in ["Macro_F1", "Accuracy"]:
                            pred = (pred > 0.5).long()
                    elif "Classification" in args.task:
                        if args.eval_metric in ["Macro_F1", "Accuracy"]:
                            pred = torch.argmax(pred, dim = 1)
                        else:
                            pred = torch.softmax(pred, dim = 1).amax(dim = 1)
                    preds = np.concatenate((preds, pred.detach().cpu().numpy()), axis = 0)
                    ys = np.concatenate((ys, y.detach().cpu().numpy()), axis = 0)
                    
                    epoch_VNs_per_layer += VNs_per_layer
                    epoch_VEs_per_layer += VEs_per_layer
                    epoch_VN_freqs_per_layer += VN_freqs_per_layer
                    num_vns += VNs_per_layer.sum().item()
                    num_ves += VEs_per_layer.sum().item()
            
            # Compute the evaluation metric
            if args.eval_metric == "AP":
                performance = AP(preds, ys)
            elif args.eval_metric == 'Macro_F1':
                performance = Macro_F1(preds, ys)
            elif args.eval_metric == "MAE":
                performance = MAE(preds, ys)
            elif args.eval_metric == "Accuracy":
                performance = Accuracy(preds, ys)
            elif args.eval_metric == "AUCROC":
                performance = AUCROC(preds, ys)
            else:
                raise NotImplementedError
            for l in range(args.num_layer):
                curr_mean_num_vns = epoch_VNs_per_layer[l].item() / num_graphs
                curr_mean_num_ves = epoch_VEs_per_layer[l].item() / num_graphs
                curr_mean_vn_freqs = (epoch_VN_freqs_per_layer[l].detach().cpu() / num_graphs).tolist()
                logger.info(f"Epoch {epoch+1} Layer {l} ({flag}) VNs:{curr_mean_num_vns}, VEs:{curr_mean_num_ves}, VN freqs:{curr_mean_vn_freqs}")
            total_mean_vn_freqs = (epoch_VN_freqs_per_layer.sum(dim = 0).detach().cpu()/num_graphs).tolist()
            logger.info(f"Epoch {epoch+1} ({flag}) VN freqs:{total_mean_vn_freqs}")
        
        results = {
            "perf": performance,
            "num_vns": num_vns,
            "num_ves": num_ves,
        }
        
        return results
    
    def run(self):
        
        args = self.args
        logger = self.logger
        data = self.data
        model = self.model
        
        best_valid_perf = 0 if args.eval_metric not in ["MAE"] else INT_MAX
        test_perf_best_valid = 0
        best_valid_epoch = 0
        for epoch in range(args.start_epoch, args.num_epoch):
            
            self.train_epoch(epoch)
            
            if (epoch+1) % args.val_dur == 0 and epoch >= args.warmup_epoch:
                val_results = self.validation(epoch)
                valid_perf = val_results["perf"]
                valid_vns = val_results["num_vns"]
                valid_ves = val_results["num_ves"]
                num_G_val = 1 if data.dataset_name in SINGLE_GRAPH_DATASETS else data.num_valid

                test_reseults = self.test(epoch)
                test_perf = test_reseults["perf"]
                test_vns = test_reseults["num_vns"]
                test_ves = test_reseults["num_ves"]
                num_G_test = 1 if data.dataset_name in SINGLE_GRAPH_DATASETS else data.num_test
            
                logger.info(f"Valid {args.eval_metric}:{valid_perf}, Valid VNs:{valid_vns/num_G_val}, Valid VEs:{valid_ves/num_G_val}, Test {args.eval_metric}:{test_perf}, Test VNs:{test_vns/num_G_test}, Test VEs:{test_ves/num_G_test}")
                
                # Check the validation best is updated
                is_lower_better = args.eval_metric in ["MAE"]
                sign = -1 if is_lower_better else 1
                if sign * (best_valid_perf - valid_perf) < 0:
                    best_valid_perf = valid_perf
                    test_perf_best_valid = test_perf
                    best_valid_epoch = epoch + 1
                    if args.save:
                        if self.scheduler is not None:
                            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': self.scheduler.state_dict()}, \
                                        f"./ckpt/{args.exp}/{args.dataset_name}/{self.file_format}_best.ckpt")
                        else:
                            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, \
                                        f"./ckpt/{args.exp}/{args.dataset_name}/{self.file_format}_best.ckpt")
                    
                logging.info(f"Best Valid {args.eval_metric} so far:{best_valid_perf}, Corresponding Test {args.eval_metric}:{test_perf_best_valid}, Epoch:{best_valid_epoch}")
            ####################################################################################
            

        logging.info(f"Best Valid {args.eval_metric}:{best_valid_perf}, Corresponding Test {args.eval_metric}:{test_perf_best_valid}, Epoch:{best_valid_epoch}")

def get_arguments() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default = "debugging", type = str, help = "experiment name")
    parser.add_argument('--log_name', default = None, type = str, help = "log file name. default: current time")
    parser.add_argument('--no_write', action = 'store_true', help = "whether to not write log file")
    parser.add_argument('--seed', default = 0, type = int, help = "random seed")
    parser.add_argument('--save', action = "store_true", help = "save best ckpt")

    dataset_args = parser.add_argument_group('dataset arguments')
    dataset_args.add_argument('--dataset_name', default = "Peptides-struct", type = str, help = "dataset name", choices = AVAILABLE_DATASETS)
    dataset_args.add_argument('--split', default = 0, type = int, help = "dataset split, for Heterophily datasets", choices = [0,1,2,3,4,5,6,7,8,9])
    dataset_args.add_argument('--dataset_dir', default = "../datasets/", type = str, help = "dataset root directory")
    dataset_args.add_argument('--task', default = "Graph Regression", type = str, help = "task type", choices = TASKS)
    dataset_args.add_argument('--multi_label', action = 'store_true', help="whether the task is multi-label classification")
    dataset_args.add_argument('--eval_metric', default = "MAE", type = str, help = "evaluation metric", choices = EVAL_METRICS)

    model_args = parser.add_argument_group('model arguments')
    model_args.add_argument('--model_name', default = "MAVN", type = str, help= "model name", choices = MODELS)
    model_args.add_argument('--base_model', default = "GatedGCN", type = str, help="base model name", choices = BASE_MODELS)
    model_args.add_argument('--dim_pe', default = '16', type = str, help = "dimension of positional encoding")
    model_args.add_argument('--dim_in', default = 256, type = int)
    model_args.add_argument('--dim_out', default = 256, type = int)
    model_args.add_argument('--dim_dot', default = 256, type = int)
    model_args.add_argument('--dim_mlp', default = 256, type = int)
    model_args.add_argument('--dim', default = -1, type = int)
    model_args.add_argument('--num_VN', default = 4, type = int, help = "number of the maximum virtual nodes")
    model_args.add_argument('--num_head', default = 1, type = int)
    model_args.add_argument('--act', default = 'GELU', type = str)
    model_args.add_argument('--normalize', default = 'None', type = str)
    model_args.add_argument('--num_layer', default = 4, type=int)
    model_args.add_argument('--num_mlp_layer', default = 1, type=int)
    model_args.add_argument('--num_pred_mlp_layer', default = 3, type = int)
    model_args.add_argument('--pred_mlp_layer_init', default = "xavier_normal", type = str)
    model_args.add_argument('--normalize_before_pred', action = 'store_true')
    model_args.add_argument('--res', action = 'store_true')
    model_args.add_argument('--readout', default = 'mean', type = str, help = "readout for prediction")
    model_args.add_argument('--norm_N', default = 'None', type = str, help = "normalization for reps_N in our module")
    model_args.add_argument('--norm_VN', default = 'None', type = str, help = "normalization for reps_VN in our module")
    model_args.add_argument('--aggr_VN', default = 'normalized_sigmoid', type = str, help = "Aggregation function for VNs")
    model_args.add_argument('--log_w', default = 1.0, type = float, help = "weight for logsoftmax")


    pe_args = parser.add_argument_group('positional encoding arguments')
    pe_args.add_argument('--pe', default = "LapPE", type = str, help="positional encoding type", choices = AVAILABLE_POSITIONAL_ENCODINGS)
    pe_args.add_argument('--pe_norm', default = "None", type = str)
    pe_args.add_argument('--num_pe', default = '10', type = str)

    train_args = parser.add_argument_group('training arguments')
    train_args.add_argument('--loss_function', default = "MAE", type = str, help = "loss function to use", choices = AVAILABLE_LOSSES)
    train_args.add_argument('--optimizer', default = "AdamW", type = str, help = "optimizer to use", choices = AVAILABLE_OPTIMIZERS)
    train_args.add_argument('--scheduler', default = "CosLRLinearWarmupRestart", type = str, help = "scheduler to use", choices = AVAILABLE_SCHEDULERS)
    train_args.add_argument('--lr_max', default=2e-3, type = float)
    train_args.add_argument('--lr_min', default = 0.0, type = float)
    train_args.add_argument('--grad_clip', default = 1.0, type = float)
    train_args.add_argument('--warmup_epoch', default = 5, type = int)
    train_args.add_argument('--restart_epoch', default = 245, type = int)
    train_args.add_argument('--restart_mult', default = 1, type = int)
    train_args.add_argument('--smoothing', default = 0.0, type = float)
    train_args.add_argument('--weight_decay', default = 0.0, type = float)
    train_args.add_argument('--dropout', default=0.1, type = float)
    train_args.add_argument('--start_epoch', default = 0, type = int)
    train_args.add_argument('--num_epoch', default=250, type = int)
    train_args.add_argument('--batch_size', default = 200, type = int)
    train_args.add_argument('--tau_start', default = 10.0, type = float)
    train_args.add_argument('--tau_end', default = 0.1, type = float)
    train_args.add_argument('--threshold', default = 0.5, type = float)
    train_args.add_argument('--tau_epoch', default = 100, type = int)
    train_args.add_argument('--weighted_loss', action = 'store_true')

    eval_args = parser.add_argument_group('evaluation arguments')
    eval_args.add_argument('--val_dur', default = 1, type = int, help = "the number of epochs per validation/test")
    eval_args.add_argument('--val_size', default = 100, type = int, help = "batch size during validation/test")
    
    args = parser.parse_args()

    if args.dim != -1:
        args.dim_in = args.dim
        args.dim_out = args.dim
        args.dim_mlp = args.dim
    if args.pe == "LapPE+RWSE":
        assert "+" in args.dim_pe, "MULTIPLE PEs REQUIRE MULTIPLE dim_pes"
        assert "+" in args.pe_norm, "MULTIPLE PEs REQUIRE MULTIPLE pe_norms"
        assert "+" in args.num_pe, "MULTIPLE PEs REQUIRE MULTIPLE num_pes"
        args.dim_pe = [int(x) for x in args.dim_pe.split("+")]
        args.num_pe = [int(x) for x in args.num_pe.split("+")]
    else:
        args.dim_pe = int(args.dim_pe)
        args.num_pe = int(args.num_pe)
    return args

def set_logger(args) -> Tuple[logging.Logger, str]:
    
    logging.Formatter.converter = lambda *args: datetime.datetime.now().timetuple()
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)
    logger.addHandler(stream_handler)

    if args.log_name is None:
        file_format = datetime.datetime.now()
    else:
        file_format = args.log_name

    if not args.no_write:
        os.makedirs(f"./ckpt/{args.exp}/{args.dataset_name}", exist_ok = True)
        os.makedirs(f"./logs/{args.exp}/{args.dataset_name}", exist_ok = True)
    else:
        file_format = None

    if not args.no_write:
        file_handler = logging.FileHandler(f"./logs/{args.exp}/{args.dataset_name}/{file_format}.log")
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    
    return logger, file_format

if __name__ == '__main__':

    args = get_arguments()

    logger, file_format = set_logger(args)
    logger.info(f"{os.getpid()}")

    try:
        exp = ExperimentManager(args, logger, file_format)
        exp.setup()
        exp.run()
    except Exception as e:
        logging.critical(e, exc_info=True)
    logging.info("Exit")
