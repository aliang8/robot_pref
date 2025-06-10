env=robomimic_lift   # ["metaworld_button-press-topdown-v2", "dmc_cheetah-run"]: env name
checkpoints_path=checkpoints  # path to save checkpoints
buffer_size=250000 # replay buffer size
frac=1.0           # fraction of data to use

python3 bc.py --config_path=configs/bc.yaml hydra/launcher=slurm --multirun