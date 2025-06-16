docker build -t robot_pref .


docker run -it \
    -v $(pwd):/workspace \
    -v /scr/shared/datasets/robot_pref:/datasets \
    --name robot_pref \
    --gpus all \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    robot_pref