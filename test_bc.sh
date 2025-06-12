data_path="/scr/shared/datasets/robot_pref/lift_sawyer/lift_sawyer.hdf5"
seed=522

python3 bc.py --config_path=configs/bc.yaml --data_path=$data_path --seed=$seed