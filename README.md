# C2DHIVFORECAST

### 1. Prerequisites

- [**Python==3.11.0**](https://www.python.org/) is required will all installed including [**PyTorch==2.1**](https://pytorch.org/get-started/previous-versions/)
- [**Pytorch Geometric**](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for Pytorch: Please match the device used when installing.
- Other dependencies are described in [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) installed including

### 2. Install

- Creating conda environment for the experiment:

```bash
conda create -n c2dhivforecast python=3.11.0 -y
conda activate c2dhivforecast
```

- Installing PyTorch, Torchvision and Pytorch Geometric depending on the device you use to run the experiment
  **For CPU version**

```bash
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

### 3. Getting started

#### Setting Datasets

(We will release soon)

#### Training and Evaluation

- Training command line:

  ```bash
  python main.py train --exp_name ${experiment_name} --root ${root} --dataset ${dataset_name} --offset ${offset} --dataset ${datasetName} --node_dim ${node_dim} --edge_dim ${edge_dim} --pooling ${pooling} --graph_block ${graph_block} --train_batch ${train_batch_size} --test_batch ${test_batch_size} --ratio ${ratio} --ignore ${ignore_feature}
  ```

  Please go through file main.py to see the detail information of all parameters.

  - Training a model from scratch:

    For example,

    To train model for data file name 'data_2008_2019.csv' in folder 'data' with transformer graph block, use topk pooling - raito 0.3, offset = 1 (mean use data 3 month predict 1 month), node embedding dim: 512, edge embedding dim: 512, batch size is 4 for test and train.

    ```bash
    python main.py train --exp_name transformer_topk3 --offset 1 --root data --dataset data_2008_2019 --node_dim 512 --edge_dim 512 --pooling topk --graph_block transformer --train_batch 4 --test_batch 4 --ratio 0.3
    ```

- Note: if you want change offset, offsetType or predictOffset, please remove Preprocessed in root.

- Evaluating command line:

  ```bash
    python main.py test --exp_name ${experiment_name} --root ${root} --dataset ${dataset_name} --offset ${offset} --dataset ${datasetName} --node_dim ${node_dim} --edge_dim ${edge_dim} --pooling ${pooling} --graph_block ${graph_block} --train_batch ${train_batch_size} --test_batch ${test_batch_size} --ratio ${ratio} --ignore ${ignore_feature}
  ```

  In order to evaluate model, setting consistently the "exp_name" equivalently to the folder name of trained model.

  For example,

  ```bash
    python main.py test --exp_name transformer_topk3 --offset 1 --root data --dataset data_2008_2019 --node_dim 512 --edge_dim 512 --pooling topk --graph_block transformer --train_batch 4 --test_batch 4 --ratio 0.3
  ```
