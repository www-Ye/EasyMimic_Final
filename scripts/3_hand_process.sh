# !/bin/bash

BASE_DIR=/path/to/datasets/human_example
OUTPUT_DIR=/path/to/datasets/human_example_processed

python 1_hand_visual_alignment.py \
  --base_dir "$BASE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --data_augmentation_type full_hand \
  --pre_interpolate_factor 0 \
  --interpolation_factor 0

python 2_hand_action_alignment.py \
  --dataset_path "$OUTPUT_DIR" \
  --coordinate_transformed true \
  --eef_type thenar_middle_joint \
  --interpolation_factor 0 \
  --backup false

python 3_generate_metadata.py \
    --dataset-path "$OUTPUT_DIR"