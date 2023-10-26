#!/bin/bash
export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_RANK
exec "$@"
