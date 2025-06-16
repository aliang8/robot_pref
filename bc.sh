#!/bin/bash

data_path="/tmp/core_datasets/expert_200/lift/demo_src_lift_task_Lift_r_Sawyer/demo.hdf5"
config_path="configs/bc.yaml"

seeds=(1)

for seed in "${seeds[@]}"; do
    echo "Launching experiment with seed $seed..."
    python3 bc.py --config_path=$config_path --data_path=$data_path --seed=$seed &
done

wait  # Wait for all background jobs to finish
echo "All experiments completed."