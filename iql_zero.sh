#!/bin/bash

env=robomimic_lift
seeds=(521 522 523)
data_quality=1.0
data_path="/scr/shared/datasets/robot_pref/lift_panda/lift_panda.hdf5"

# Run for three seeds
for seed in "${seeds[@]}"; do
    echo "Running seed $seed"
    python3 iql.py --config=configs/iql.yaml --env=$env \
    --data_quality=$data_quality --seed=$seed \
    --data_path=$data_path --trivial_reward=1
done 

wait  # Wait for all background jobs to finish
echo "All experiments completed."