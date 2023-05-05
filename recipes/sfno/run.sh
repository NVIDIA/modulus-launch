#!/bin/bash

#pip uninstall -y nvidia-modulus
#cd ../../../modulus
#pip install -e .
#cd ../modulus-launch/recipes/sfno

export WANDB_MODE=disabled

python -u train.py --yaml_config=./config/sfnonet.yaml --run_num=test --config=base_config --enable_synthetic_data
