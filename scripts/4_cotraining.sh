python scripts/gr00t_finetune.py \
  --dataset-path /path/to/datasets/robot_example \
    /path/to/datasets/human_example_processed \
  --embodiment-tags lerobot_plus human_hand \
  --no-balance-dataset-weights \
  --no-individual-transforms \
  --num-gpus 1 \
  --output-dir ./checkpoints/example_cotraining \
  --max-steps 5000 \
  --save-steps 5000 \
  --data-config so100_plus_abs_min_max \
  --video-backend torchvision_av \
  --no-tune-llm \
  --no-tune-visual \
  --tune-projector \
  --tune-diffusion-model \
  --report-to tensorboard