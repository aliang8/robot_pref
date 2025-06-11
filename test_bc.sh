env=robomimic_lift   # ["metaworld_button-press-topdown-v2", "dmc_cheetah-run"]: env name
data_path="/scr/shared/datasets/robot_pref/lift_sawyer/lift_sawyer.hdf5"
seed=42

checkpoints_path=checkpoints  # path to save checkpoints
buffer_size=250000 # replay buffer size
frac=1.0           # fraction of data to use

python3 bc.py --config_path=configs/bc.yaml --data_path=$data_path --seed=$seed