env=robomimic_square   # ["metaworld_button-press-topdown-v2", "dmc_cheetah-run"]: env name
seed=42             # random seed
batch_size=256      # batch size for training
eval_freq=1000      # evaluation frequency
n_episodes=10       # number of episodes for evaluation
checkpoints_path=checkpoints  # path to save checkpoints
buffer_size=200027 # replay buffer size
frac=1.0           # fraction of data to use
normalize=True      # whether to normalize observations

python3 bc.py --config_path=configs/bc.yaml --env=$env \
--seed=$seed --eval_freq=$eval_freq --n_episodes=$n_episodes \
--checkpoints_path=$checkpoints_path --batch_size=$batch_size \
--buffer_size=$buffer_size --frac=$frac --normalize=$normalize