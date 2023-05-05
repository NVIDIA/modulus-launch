#!/bin/bash

nvidia-smi
git config --global --add safe.directory /code/modulus-launch
#cp /secrets/.netrc ~/

export TOTALGPU=$((NGC_GPUS_PER_NODE * NGC_ARRAY_SIZE))
readonly _config_name='base_config'
readonly yaml_config='./config/sfnonet_mn.yaml'

#export WANDB_MODE=disabled

bcprun --nnodes $NGC_ARRAY_SIZE --npernode $NGC_GPUS_PER_NODE --workdir /code/modulus-launch/recipes/sfno --env WANDB_MODE=offline --cmd "python -u train.py --yaml_config=${yaml_config} --run_num=ngpu${TOTALGPU} --config=${_config_name} --cuda_graph_mode=none"
