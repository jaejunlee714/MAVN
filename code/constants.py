############################### ABOUT DATASETS ############################### 

AVAILABLE_DATASETS = ["Peptides-func", "Peptides-struct", "PascalVOC-SP", "COCO-SP",
                      "roman_empire", "amazon_ratings", "minesweeper", "tolokers", "questions"]
LRGB_GRAPH_DATASETS = ["Peptides-func", "Peptides-struct"]
LRGB_NODE_DATASETS = ["PascalVOC-SP", "COCO-SP"]
HETEROPHILY_DATASETS = ["roman_empire", "amazon_ratings", "minesweeper", "tolokers", "questions"]

SINGLE_GRAPH_DATASETS = [] + HETEROPHILY_DATASETS

TASKS = ["Graph Classification", "Graph Regression", "Node Classification"]
EVAL_METRICS = ["AP", "Macro_F1", "MAE", "AUCROC", "Accuracy"]
DATASET_TASK_DICT = {
    "Peptides-func": ["Graph Classification", "Graph Regression"],
    "Peptides-struct": ["Graph Classification", "Graph Regression"],
    "PascalVOC-SP": ["Node Classification"],
    "COCO-SP": ["Node Classification"],
    
    "roman_empire": ["Node Classification"],
    "amazon_ratings": ["Node Classification"],
    "minesweeper": ["Node Classification"],
    "tolokers": ["Node Classification"],
    "questions": ["Node Classification"]
}

NODE_FEAT_TYPE_DICT = {
    "Peptides-func": "Categorical",
    "Peptides-struct": "Categorical",
    "PascalVOC-SP": "Linear",
    "COCO-SP": "Linear",
    
    "roman_empire": "LinearDrop",
    "amazon_ratings": "None",
    "minesweeper": "None",
    "tolokers": "None",
    "questions": "LinearDrop"
}

EDGE_FEAT_TYPE_DICT = {
    "Peptides-func": "Categorical",
    "Peptides-struct": "Categorical",
    "PascalVOC-SP": "Linear",
    "COCO-SP": "Linear",
    
    "roman_empire": "Linear",
    "amazon_ratings": "Linear",
    "minesweeper": "Linear",
    "tolokers": "Linear",
    "questions": "Linear"
}

#############################################################################

AVAILABLE_LOSSES = ["BCE", "CE", "MAE"]
AVAILABLE_OPTIMIZERS = ["AdamW", "Adam"]
AVAILABLE_SCHEDULERS = ["CosLRLinearWarmupRestart", "None"]
AVAILABLE_POSITIONAL_ENCODINGS = ["LapPE", "RWSE", "LapPE+RWSE", "None"]

MODELS = ["GNN", "MAVN"]
BASE_MODELS = ["GCN-LRGB", "GINE-LRGB", "GatedGCN-LRGB", "GCN-TunedGNN", "GraphSAGE-TunedGNN", "GAT-TunedGNN", "GCN-Hete", "GraphSAGE-Hete", "GAT-Hete"]

INT_MAX = 2147483647