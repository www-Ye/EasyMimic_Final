#!/usr/bin/env bash
set -euo pipefail

# 默认 Python 解释器
PYTHON_BIN="${PYTHON_BIN:-python}"

# 默认参数（与 hand_api/2_update_handdataset.py 对齐）
DATASET_PATH="/home/admin123/projects_lerobot/Easymimic/lerobot-joycon_plus/datasets/pull_push/human/human_25_0904_clear"
CONFIG_PATH="/home/admin123/projects_lerobot/Easymimic/lerobot-joycon_plus/lerobot/configs/robot/so100_plus_joycon_single_new.yaml"
COORDINATE_TRANSFORMED="true"
EEF_TYPE="thenar_middle_joint"
INTERPOLATION_FACTOR="0"
BACKUP="false"
START_FROM=""
MAX_EPISODES=""
SKIP_FAILED="true"
CHUNK_NAME="chunk-000"

print_help() {
  cat <<EOF

EOF
}

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-path)
      DATASET_PATH="$2"; shift 2 ;;
    --config-path)
      CONFIG_PATH="$2"; shift 2 ;;
    --coordinate-transformed)
      COORDINATE_TRANSFORMED="$2"; shift 2 ;;
    --eef-type)
      EEF_TYPE="$2"; shift 2 ;;
    --interpolation-factor)
      INTERPOLATION_FACTOR="$2"; shift 2 ;;
    --backup)
      BACKUP="true"; shift 1 ;;
    --start-from)
      START_FROM="$2"; shift 2 ;;
    --max-episodes)
      MAX_EPISODES="$2"; shift 2 ;;
    --skip-failed)
      SKIP_FAILED="true"; shift 1 ;;
    --python-bin)
      PYTHON_BIN="$2"; shift 2 ;;
    --chunk-name)
      CHUNK_NAME="$2"; shift 2 ;;
    -h|--help)
      print_help; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      print_help
      exit 1 ;;
  esac
done

if [[ -z "${DATASET_PATH}" ]]; then
  echo "[Error] --dataset-path 为必填参数" >&2
  print_help
  exit 1
fi

UPDATE_ARGS=(
  --dataset_path "${DATASET_PATH}"
  --config_path "${CONFIG_PATH}"
  --coordinate_transformed "${COORDINATE_TRANSFORMED}"
  --eef_type "${EEF_TYPE}"
  --interpolation_factor "${INTERPOLATION_FACTOR}"
)

if [[ "${BACKUP}" == "true" ]]; then
  UPDATE_ARGS+=(--backup)
fi

if [[ -n "${START_FROM}" ]]; then
  UPDATE_ARGS+=(--start_from "${START_FROM}")
fi

if [[ -n "${MAX_EPISODES}" ]]; then
  UPDATE_ARGS+=(--max_episodes "${MAX_EPISODES}")
fi

if [[ "${SKIP_FAILED}" == "true" ]]; then
  UPDATE_ARGS+=(--skip_failed)
fi

echo "==> Step 1/2: Running 2_update_handdataset.py ..."
"${PYTHON_BIN}" /home/admin123/projects_lerobot/Easymimic/hand_api/2_update_handdataset.py "${UPDATE_ARGS[@]}"

echo "==> Step 2/2: Running 3_generate_metadata.py ..."
"${PYTHON_BIN}" /home/admin123/projects_lerobot/Easymimic/hand_api/3_generate_metadata.py \
  --dataset-path "${DATASET_PATH}" \
  --chunk-name "${CHUNK_NAME}"

echo "==> All done."