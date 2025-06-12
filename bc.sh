#!/bin/bash

data_path="/scr/shared/datasets/robot_pref/lift_sawyer/lift_sawyer.hdf5"
seeds=(521 522 523)
config_path="configs/bc.yaml"

for seed in "${seeds[@]}"; do
    echo "Launching experiment with seed $seed..."
    python3 bc.py --config_path=$config_path --data_path=$data_path --seed=$seed &
done

wait  # Wait for all background jobs to finish
echo "All experiments completed."