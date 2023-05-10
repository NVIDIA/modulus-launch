#!/bin/bash

#cd ../../../modulus
#pip install .
#cd ../modulus-launch/recipes/sfno

export WANDB_MODE=disabled

python -u train.py --yaml_config=./config/sfnonet.yaml --run_num=test --config=base_config --enable_synthetic_data
