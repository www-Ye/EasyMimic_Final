#!/bin/bash

# --- 配置路径 ---
BASE_PATH="/home/admin123/projects/lerobot-plus/lerobot-joycon_plus_data/datasets/pick_put_hand_v1_reset"
IMG_PATH="${BASE_PATH}/images/observation.images.webcam"
NEW_BASE_PATH="/home/admin123/projects/lerobot-plus/lerobot-joycon_plus_data/datasets/pick_put_hand_v1_reset_new"
TASK_NAME="Pick up the object and put it into the bowl."
PYTHON_SCRIPT="hand_single_process_api.py" # 指向你的Python脚本

# --- 创建必要的输出目录 ---
mkdir -p "${NEW_BASE_PATH}"
mkdir -p "${NEW_BASE_PATH}/meta"
mkdir -p "${NEW_BASE_PATH}/data/chunk-000"
mkdir -p "${NEW_BASE_PATH}/videos/chunk-000/observation.images.webcam"

echo "Created output directory structure in: ${NEW_BASE_PATH}"

# --- 初始化计数器 ---
CNT=0 # Counts successfully processed episodes (length > 0)
TOTAL_EP_LENS=0 # Accumulates frames from successful episodes
OUTPUT_EPISODE_INDEX=0 # Index for sequential output naming

# --- 查找并处理每个episode文件夹 ---
# 使用 find 查找，确保能处理包含空格等特殊字符的路径，并排序
find "${IMG_PATH}" -mindepth 1 -maxdepth 1 -type d -name 'episode_*' | sort | while IFS= read -r input_file_path; do
    if [ -d "$input_file_path" ]; then
        input_episode_basename=$(basename "$input_file_path")
        echo "-----------------------------------------------------"
        echo "Input Episode: ${input_episode_basename} (Path: ${input_file_path})"

        # 生成顺序的输出 episode 名称
        output_episode_name=$(printf "episode_%06d" $OUTPUT_EPISODE_INDEX)
        echo "Target Output Name: ${output_episode_name}"
        echo "-----------------------------------------------------"

        # 执行 Python 脚本处理单个 episode，传递顺序输出名称
        # 添加 --enable_visualization 如果需要可视化
        output=$(python "${PYTHON_SCRIPT}" \
                        --file_path "$input_file_path" \
                        --new_base_path "$NEW_BASE_PATH" \
                        --episode_name "$output_episode_name" \
                        --enable_visualization) # 根据需要添加此行

        # 检查 Python 脚本是否因严重错误退出 (非零状态码)
        exit_status=$?
        if [ $exit_status -ne 0 ]; then
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            echo "FATAL ERROR processing input ${input_episode_basename} (Target: ${output_episode_name}). Python script exited with status ${exit_status}."
            echo "Python script output (may contain critical error details):"
            echo "${output}"
            echo "STOPPING the entire process."
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            exit 1 # 停止整个 shell 脚本
        fi

        # 如果 Python 脚本成功完成 (exit_status 为 0)，提取 episode length
        # --- 修正：使用 grep -e ---
        ep_lens=$(echo "$output" | grep -e '---EPISODE_LENGTH---:' | tail -n 1 | cut -d ':' -f 2)
        # -------------------------

        # 检查是否成功提取到 ep_lens 并且是数字
        if [[ "$ep_lens" =~ ^[0-9]+$ ]]; then
            if [ "$ep_lens" -gt 0 ]; then
                # 成功处理并得到有效帧数
                echo "Successfully processed input ${input_episode_basename} as ${output_episode_name}. Length: ${ep_lens}"
                CNT=$((CNT + 1))
                TOTAL_EP_LENS=$((TOTAL_EP_LENS + ep_lens))
                # 只有成功处理（长度>0）后，才增加下一个输出的序列号
                OUTPUT_EPISODE_INDEX=$((OUTPUT_EPISODE_INDEX + 1))
            else
                # Python 脚本正常完成，但返回长度为 0 (表示跳过)
                echo "Skipped input ${input_episode_basename} (Returned length 0). Output index ${OUTPUT_EPISODE_INDEX} will be reused."
            fi
        else
            # Python 脚本正常完成，但未能从输出中提取长度 (这不应该发生)
            echo "Warning: Could not extract valid episode length for input ${input_episode_basename} from script output, even though script exited successfully (status 0)."
            echo "Please check the Python script output:"
            echo "$output"
            # 决定如何处理，这里选择警告并继续，但序列号可能不增加
            echo "Output index ${OUTPUT_EPISODE_INDEX} might be reused or skipped incorrectly."
        fi
        echo # 添加空行以便区分不同episode的输出
    fi
done

echo "====================================================="
echo "Finished processing all input folders (or stopped due to fatal error)."
echo "Total episodes successfully processed and saved (length > 0): ${CNT}"
echo "Total frames accumulated from saved episodes: ${TOTAL_EP_LENS}"
echo "Next output episode index would be: ${OUTPUT_EPISODE_INDEX}"
echo "====================================================="

# --- 更新 meta/info.json ---
# 只有在脚本正常完成所有处理时才执行
BASE_INFO_JSON="${BASE_PATH}/meta/info.json" # 仍然从原始数据集读取模板
NEW_INFO_JSON="${NEW_BASE_PATH}/meta/info.json"

if [ -f "$BASE_INFO_JSON" ]; then
    echo "Updating ${NEW_INFO_JSON}..."
    if command -v jq &> /dev/null; then
        # 使用最终的 CNT (成功处理的 episode 数) 更新 JSON
        jq \
          --argjson cnt "$CNT" \
          --argjson total_len "$TOTAL_EP_LENS" \
          '.total_episodes = $cnt | .total_frames = $total_len | .total_videos = $cnt' \
          "$BASE_INFO_JSON" > "$NEW_INFO_JSON"
        echo "info.json updated successfully with final counts (Episodes: ${CNT}, Frames: ${TOTAL_EP_LENS})."
    else
        echo "Warning: 'jq' command not found. Cannot update info.json automatically."
    fi
else
    echo "Warning: Base info.json not found at ${BASE_INFO_JSON}. Cannot create updated info.json."
fi

echo "Script finished."