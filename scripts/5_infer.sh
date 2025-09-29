python scripts/inference_service.py --server \
    --model_path ./checkpoints/example_cotraining \
    --data_config so100_plus_abs_min_max \
    --denoising_steps 4 \
    --embodiment_tag lerobot_plus \
    --video_mode dual

python lerobot/scripts/control_robot_gr00t.py record \
  --robot-path lerobot/configs/robot/so100_plus_joycon_single_new.yaml \
  --fps 30 \
  --tags so100_plus tutorial eval \
  --warmup-time-s 5 \
  --episode-time-s 40 \
  --reset-time-s 6 \
  --num-episodes 10 \
  --push-to-hub 0 \
  --local-files-only 1 \
  --root /path/to/datasets/eval_example \
  --use-gr00t True \
  --host "localhost" \
  --port 5555 \
  --lang-instruction "[Robot] Pick up the object and put it into the bowl." \
  --repo-id task/eval_pick_put \
  --single-task eval_pick_put