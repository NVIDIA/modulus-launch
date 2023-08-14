This code replicates the numerical results of the paper [`Learning Reduced-Order Models for Cardiovascular Simulations with Graph Neural Networks`](https://arxiv.org/abs/2303.07310) (Pegolotti et al, 2023).

## Download dataset
To download the dataset (vtp simulation files):
```bash
cd raw_dataset
bash download_dataset.sh
```

## Generate graphs
To generate graphs out of the vtp files:
```bash
python generate_graphs.py
```
This will create a new `graph` folder `in raw_dataset`.

## Train the model
To train the model:
```bash
python train.py
```
The training parameters can be modified in `config.yaml`. An important parameter
is `training.geometries`, which can take the values `healthy`, `pathological`, 
`mixed`. These `healthy` and `pathological` refer to the geometries used in
Section 5.1 and 5.2 of the paper; `mixed` considers all geometries.

## Evaluate Model
To perform inference on a given model:
```bash
python inference.py
```
The name of the model needs to be specified in `config.yaml`.
