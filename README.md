# Learn When and Where to Connect: Adaptive Virtual Nodes for Dynamic Message Passing on Graphs

This is the official code of "Learn When and Where to Connect: Adaptive Virtual Nodes for Dynamic Message Passing on Graphs"\
**Published in the 32nd SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD 2026 Research Track).**

All codes are written by [Jaejun Lee](https://jaejunlee714.github.io/) (jjlee98@kaist.ac.kr).\
If you use this code, please cite our paper.

```bibtex
@inproceedings{mavn,
	author={Jaejun Lee and Joyce Jiyoung Whang},
	title={{Learn When and Where to Connect}: Adaptive Virtual Nodes for Dynamic Message Passing on Graphs},
	booktitle={Proceedings of the 32nd SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
	year={2026},
	pages={}
}
```
<!--
```bibtex
@inproceedings{mavn,
  author={Jaejun Lee and Joyce Jiyoung Whang},
  title={{Learn When and Where to Connect}: Adaptive Virtual Nodes for Dynamic Message Passing on Graphs},
  year={2026},
	journal={},
	doi={},
}
```
-->

## Requirements

We used python 3.9.19 and PyTorch 2.0.1 with cudatoolkit 11.7.

You can install all requirements (except python) with:

```setup
pip install -r requirements.txt
```

## Downloading Datasets

You can download the datasets from the following links:

Peptides-func: [Link](https://www.dropbox.com/s/ycsq37q8sxs1ou8/peptidesfunc.zip?dl=1)\
PascalVOC-SP: [Link](https://www.dropbox.com/s/8x722ai272wqwl4/pascalvocsp.zip?dl=1)\
minesweeper: [Link](https://github.com/yandex-research/heterophilous-graphs/raw/refs/heads/main/data/minesweeper.npz)\
tolokers: [Link](https://github.com/yandex-research/heterophilous-graphs/raw/refs/heads/main/data/tolokers.npz)

The links for Peptides-func and PascalVOC-SP are sourced from [Torch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/lrgb.html#LRGBDataset), while the minesweeper and tolokers links come from their original [GitHub repository](https://github.com/yandex-research/heterophilous-graphs).

### Dataset Setup Instructions

First, create the main dataset directory, `./datasets/`.

Then follow the steps below for each dataset:

- Peptides-func
1. Unzip the downloaded file
2. Rename `val.pt` to `valid.pt`
3. Create `./datasets/Peptides-func/` and move `train.pt`, `valid.pt`, and `test.pt`.

- PascalVOC-SP
1. Unzip the downloaded file
2. Rename val.pickle to valid.pickle
3. Create `./datasets/PascalVOC-SP/` and move `train.pickle`, `valid.pickle`, and `test.pickle`.

- minesweeper
  - Create `./datasets/minesweeper/` and move `minesweeper.npz`.

- tolokers
  - Create  `./datasets/tolokers/` and move `tolokers.npz`.

## Training
We used NVIDIA RTX 2080 Ti, NVIDIA RTX 3090 and NVIDIA RTX A6000, depending on the dataset being used.\
Two different implementations were used to train models on heterophilic graph datasets, as detailed in Appendix B.

Implementation from [1]:train.py\
Implementation from [2]:train_v2.py

[1] Luo et al., "Classic GNNs are Strong Baselines: Reassessing GNNs for Node Classification", NeurIPS 2024 Track on Datasets and Benchmarks.\
[2] Platanov et al., "A critical look at the evaluation of GNNs under heterophily: Are we really making progress?", ICLR 2023.

### Peptides-func

This dataset was trained on NVIDIA RTX 3090.

To reproduce the results of MAVN reported in our paper, use the following command from the `./code/` directory:

```python
python train.py --exp 'KDD2026' --log_name 'MAVN' --seed 0 --dataset_name 'Peptides-func' --task 'Graph Classification' --eval_metric 'AP' --model_name 'MAVN' --base_model 'GCN-LRGB' --dim_pe '8+28' --dim 245 --dim_dot 16 --num_VN 12 --num_head 4 --act 'GELU' --normalize 'BatchNorm' --num_layer 6 --num_mlp_layer 1 --num_pred_mlp_layer 2 --norm_N 'GraphNorm' --norm_VN 'LayerNorm' --aggr_VN 'normalized_sigmoid' --log_w 1.0 --pe 'LapPE+RWSE' --pe_norm 'None+Batch' --num_pe '150+20' --loss_function 'BCE' --optimizer 'AdamW' --lr_max 2e-3 --grad_clip 1.0 --warmup_epoch 15 --restart_epoch 735 --dropout 0.2 --num_epoch 750 --batch_size 200 --tau_end 0.1 --tau_epoch 300 --multi_label --normalize_before_pred 
```

**Note**: We used four random seeds for evaluation: 0, 1, 2, and 3.

### PascalVOC-SP

This dataset was trained on NVIDIA RTX A6000.

To reproduce the results of MAVN reported in our paper, use the following command from the `./code/` directory:

```python
python train.py --exp 'KDD2026' --log_name 'MAVN' --seed 0 --dataset_name 'PascalVOC-SP' --task 'Node Classification' --eval_metric 'Macro_F1' --model_name 'MAVN' --base_model 'GatedGCN-LRGB' --dim_pe '0' --dim 90 --dim_dot 32 --num_VN 24 --num_head 1 --act 'GELU' --normalize 'BatchNorm' --num_layer 10 --num_mlp_layer 1 --num_pred_mlp_layer 2 --norm_N 'GraphNorm' --norm_VN 'None' --aggr_VN 'normalized_sigmoid' --log_w 1.0 --pe 'None' --pe_norm 'None' --num_pe '0' --loss_function 'CE' --optimizer 'AdamW' --lr_max 3e-3 --grad_clip 0.0 --warmup_epoch 10 --restart_epoch 190 --dropout 0.1 --num_epoch 200 --batch_size 50 --tau_end 1.0 --tau_epoch 100 --weighted_loss --normalize_before_pred 
```

**Note**: We used four random seeds for evaluation: 0, 1, 2, and 3.

### minesweeper

This dataset was trained on NVIDIA RTX 2080 Ti.

To reproduce the results of MAVN reported in our paper, use the following command from the `./code/` directory:

```python
python train.py --exp 'KDD2026' --log_name 'MAVN' --seed 0 --dataset_name 'minesweeper' --split 0 --task 'Node Classification' --eval_metric 'AUCROC' --model_name 'MAVN' --base_model 'GraphSAGE-TunedGNN' --dim_pe '0' --dim 64 --dim_dot 64 --num_VN 80 --num_head 1 --act 'ReLU' --normalize 'BatchNorm' --num_layer 15 --num_mlp_layer 1 --num_pred_mlp_layer 1 --norm_N 'GraphNorm' --norm_VN 'None' --aggr_VN 'normalized_sigmoid' --log_w 0.2 --pe 'None' --pe_norm 'None' --num_pe '0' --loss_function 'CE' --optimizer 'Adam' --lr_max 5e-2 --grad_clip 0.0 --warmup_epoch 300 --restart_epoch 2700 --dropout 0.2 --num_epoch 3000 --batch_size -1 --tau_end 0.1 --tau_epoch 1500 --pred_mlp_layer_init 'default' --res
```

**Note**: We used 10 data splits for evaluation: 0 through 9.

### tolokers

This dataset was trained on NVIDIA RTX 2080 Ti.

To reproduce the results of MAVN reported in our paper, use the following command from the `./code/` directory:

```python
python train_v2.py --exp 'KDD2026' --log_name 'MAVN' --seed 0 --dataset_name 'tolokers' --split 0 --task 'Node Classification' --eval_metric 'AUCROC' --model_name 'MAVN' --base_model 'GAT-Hete' --dim_pe '0' --dim 32 --dim_dot 96 --num_VN 72 --num_head 4 --act 'GELU' --normalize 'LayerNorm' --num_layer 6 --num_mlp_layer 1 --num_pred_mlp_layer 1 --norm_N 'LayerNorm' --norm_VN 'LayerNorm' --aggr_VN 'sigmoid' --log_w 0.1 --pe 'None' --pe_norm 'None' --num_pe '0' --loss_function 'BCE' --optimizer 'AdamW' --lr_max 1e-2 --grad_clip 0.0 --warmup_epoch 100 --restart_epoch 1900 --dropout 0.4 --num_epoch 2000 --batch_size -1 --tau_end 0.1 --tau_epoch 1000 --normalize_before_pred 
```

**Note**: We used 10 data splits for evaluation: 0 through 9.

**Please refer to [Setup_MAVN_KDD2026.pdf](https://github.com/bdi-lab/MAVN/blob/main/Setup_MAVN_KDD2026.pdf) for the hyperparameter settings for the other datasets.**

## Licenses

All codes in this repository is licensed under the [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.
