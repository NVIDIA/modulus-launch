#!/bin/bash

ngc batch run \
    --name "modulus_sfno_multi-node,ml-model.earth2" \
    --priority HIGH \
    --preempt RUNONCE \
    --min-timeslice 1H \
    --total-runtime 7d \
    --array-type "PYTORCH" --replicas "2" \
    --port 8888 \
    --result /result \
    --ace nv-us-west-3 \
    --image "nvcr.io/nvidian/earth2/modulus-launch-sfno:0.0" \
    --org nvidian \
    --team earth2 \
    --datasetid 1601932:/ngc_era5_data \
    --datasetid 1605637:/ifs \
    --workspace sfno_modulus:/code:RW \
    --instance dgxa100.80g.8.norm \
    --commandline "/bin/bash /code/run_mn.sh"
