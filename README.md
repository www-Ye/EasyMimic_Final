# EasyMimic: 基于 GR00T 的人-机协同训练框架

EasyMimic 是一个基于 NVIDIA Isaac GR00T 和 LeRobot 的机器人模仿学习框架。它提供了一个从数据采集、处理、协同训练到推理部署的完整解决方案，旨在让机器人通过观察人类演示来学习复杂任务。

## 快速开始：完整流程

本节将指导您完成从数据采集到模型部署的完整流程，每个步骤都有详细的说明和实际示例。

### 1. 人类手部数据采集

首先，我们需要采集人类执行任务的手部视频数据：

```bash
python lerobot/scripts/control_robot_reset.py record \
    --robot-path lerobot/configs/robot/so100_plus_joycon_single_new.yaml \
    --fps 30 \
    --tags so100 tutorial \
    --warmup-time-s 5 \
    --episode-time-s 6 \
    --reset-time-s 4 \
    --num-episodes 20 \
    --push-to-hub 0 \
    --local-files-only 1 \
    --root /path/to/datasets/human_example \
    --repo-id task/pick \
    --single-task "[Human] Pick up the object and put it into the bowl." \
    --resume 1
```

关键参数说明：
- **robot-path**: 指定机器人配置文件路径，这里使用单个Joy-Con遥控器配置
- **num-episodes**: 采集的总轨迹数
- **episode-time-s**: 每段轨迹的持续时间（秒）
- **root**: 保存数据的根目录
- **single-task**: 任务描述，以"[Human]"开头表示人类演示

### 2. 机器人遥控数据采集

接下来，通过Joy-Con遥控器控制机器人采集演示数据：

```bash
python lerobot/scripts/control_robot_reset.py record \
    --robot-path lerobot/configs/robot/so100_plus_joycon_single_new.yaml \
    --fps 30 \
    --tags so100 tutorial \
    --warmup-time-s 5 \
    --episode-time-s 18 \
    --reset-time-s 6 \
    --num-episodes 20 \
    --push-to-hub 0 \
    --local-files-only 1 \
    --root /path/to/datasets/robot_example \
    --repo-id task/pick_put \
    --single-task "[Robot] Pick up the object and put it into the bowl." \
    --resume 1
```

关键参数说明：
- **episode-time-s**: 通常比人类演示时间更长，这里设为18秒
- **single-task**: 以"[Robot]"开头表示机器人演示

### 3. 手部数据处理

对采集的人类手部数据进行处理，转换为机器人可理解的格式：

```bash
# 第1步: 视觉对齐处理
python 1_hand_visual_alignment.py \
  --base_dir "/path/to/datasets/human_example" \
  --output_dir "/path/to/datasets/human_example_processed" \
  --data_augmentation_type full_hand \
  --pre_interpolate_factor 0 \
  --interpolation_factor 0

# 第2步: 动作对齐处理
python 2_hand_action_alignment.py \
  --dataset_path "/path/to/datasets/human_example_processed" \
  --coordinate_transformed true \
  --eef_type thenar_middle_joint \
  --interpolation_factor 0 \
  --backup false

# 第3步: 生成元数据
python 3_generate_metadata.py \
  --dataset-path "/path/to/datasets/human_example_processed"
```

关键步骤说明：
- **视觉对齐**: 预处理视频数据，确定手部帧
- **动作对齐**: 提取3D手部姿态，通过SLAM进行位姿估计，计算末端执行器轨迹
- **元数据生成**: 计算数据集统计信息，用于后续训练归一化

重要参数说明：
- **data_augmentation_type**: `full_hand`表示使用完整手部模型
- **eef_type**: `thenar_middle_joint`表示以虎口中点为末端执行器
- **coordinate_transformed**: `true`表示转换至机器人基坐标系

### 4. 协同训练模型

将机器人数据和处理后的人类手部数据一起进行联合训练：

```bash
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
```

关键参数说明：
- **dataset-path**: 列出所有参与训练的数据集路径（机器人数据和人类数据）
- **embodiment-tags**: 为每个数据集指定对应的身体化标签，顺序与dataset-path对应
- **no-tune-llm/no-tune-visual**: 冻结大型骨干网络以节省资源
- **tune-projector/tune-diffusion-model**: 只微调特定组件
- **max-steps**: 训练步数

### 5. 模型推理部署

最后，将训练好的模型部署到机器人上进行在线推理：

```bash
# 第1步: 启动推理服务
python scripts/inference_service.py --server \
    --model_path ./checkpoints/example_cotraining \
    --data_config so100_plus_abs_min_max \
    --denoising_steps 4 \
    --embodiment_tag lerobot_plus \
    --video_mode dual

# 第2步: 启动机器人控制客户端
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
```

关键参数说明：
- **推理服务**:
  - **model_path**: 训练好的模型路径
  - **denoising_steps**: 扩散模型降噪步数，影响推理速度和质量
  - **video_mode**: `dual`表示使用两个相机视角

- **控制客户端**:
  - **use-gr00t**: 设为`True`启用GR00T模型控制
  - **host/port**: 连接到推理服务的地址和端口
  - **lang-instruction**: 指示模型执行的任务


## 数据集格式

EasyMimic 遵循 LeRobot 数据格式规范，每个数据集包含：

- **images/observation.images.webcam/**: 包含多个 episode_XXXXXX 文件夹，每个文件夹存放一个轨迹的视频帧
- **data/chunk-000/**: 包含 .parquet 文件，存储对应每个轨迹的状态和动作序列
- **metadata.json**: 数据集元数据，包括状态和动作的统计信息（均值、标准差、最大/最小值）
