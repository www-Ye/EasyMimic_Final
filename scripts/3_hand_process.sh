# !/bin/bash

BASE_DIR=/path/to/datasets/human_example
OUTPUT_DIR=/path/to/datasets/human_example_processed

# step 1: 生成visual alignment 数据
python 1_hand_visual_alignment.py \
  --base_dir "$BASE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --data_augmentation_type full_hand \
  --pre_interpolate_factor 0 \
  --interpolation_factor 0

# step 2: 生成 action alignment 数据
python 2_hand_action_alignment.py \
  --dataset_path "$OUTPUT_DIR" \
  --coordinate_transformed true \
  --eef_type thenar_middle_joint \
  --interpolation_factor 0 \
  --backup false

# step 3: 生成 metadata.json
python 3_generate_metadata.py \
    --dataset-path "$OUTPUT_DIR"