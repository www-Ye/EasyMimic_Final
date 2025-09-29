import subprocess
import shlex # 用于安全地处理命令行参数
import os
import json # 用于解析 ffprobe 的输出

def get_video_dimensions(input_path):
    """使用 ffprobe 获取视频的宽度和高度"""
    try:
        # 构造 ffprobe 命令
        # -v quiet: 安静模式，减少输出
        # -print_format json: 以 JSON 格式输出信息
        # -show_streams: 显示流信息（包含视频流）
        # -select_streams v:0: 只选择第一个视频流
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-select_streams", "v:0",
            input_path
        ]

        # 执行命令并获取输出
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        # 提取宽度和高度
        if 'streams' in data and len(data['streams']) > 0:
            video_stream = data['streams'][0]
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            return width, height
        else:
            print("错误：无法从 ffprobe 输出中找到视频流信息。")
            return None, None

    except FileNotFoundError:
        print("错误：找不到 ffprobe 命令。请确保 ffmpeg (包含 ffprobe) 已安装并在系统 PATH 中。")
        return None, None
    except subprocess.CalledProcessError as e:
        print(f"错误：执行 ffprobe 失败: {e}")
        print(f"ffprobe stderr: {e.stderr}")
        return None, None
    except Exception as e:
        print(f"获取视频尺寸时发生未知错误: {e}")
        return None, None


def crop_video_center_ffmpeg(input_path, output_path, target_width, target_height):
    """
    使用 ffmpeg 命令行从视频中心裁剪出指定尺寸的新视频。

    Args:
        input_path (str): 输入视频文件路径。
        output_path (str): 输出视频文件路径。
        target_width (int): 目标宽度。
        target_height (int): 目标高度。
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_path):
            print(f"错误：输入文件不存在 {input_path}")
            return

        # 获取原始视频尺寸 (可选，但推荐用于检查)
        original_width, original_height = get_video_dimensions(input_path)
        if original_width is None or original_height is None:
            print("无法获取原始视频尺寸，无法继续。")
            return

        print(f"原始视频尺寸: {original_width}x{original_height}")
        print(f"目标裁剪尺寸: {target_width}x{target_height}")

        # 检查目标尺寸是否小于原始尺寸
        if target_width > original_width or target_height > original_height:
            print(f"错误：目标尺寸 ({target_width}x{target_height}) "
                  f"大于原始视频尺寸 ({original_width}x{original_height})。无法裁剪。")
            return

        # 构建 ffmpeg 命令
        # -i: 输入文件
        # -vf: 视频滤镜 (video filter)
        #   crop=w:h:x:y : 裁剪滤镜
        #     w: 输出宽度 (target_width)
        #     h: 输出高度 (target_height)
        #     x: 裁剪区域左上角的 x 坐标。'(in_w-out_w)/2' 表示输入宽度减输出宽度除以2，即水平居中
        #     y: 裁剪区域左上角的 y 坐标。'(in_h-out_h)/2' 表示输入高度减输出高度除以2，即垂直居中
        # -c:v libx264: 使用 H.264 视频编码器
        # -crf 23: H.264 的质量因子 (Constant Rate Factor)，数值越小质量越高，文件越大。18-28 是常见范围。
        # -preset medium: 编码速度与压缩率的平衡，可选 ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
        # -c:a copy: 直接复制音频流，不做重新编码（更快，无质量损失）
        # -y: 如果输出文件已存在，自动覆盖

        # 注意：ffmpeg 的 filtergraph 语法中，可以直接使用 in_w, in_h, out_w, out_h 变量
        filter_command = f"crop=w={target_width}:h={target_height}:x=(in_w-{target_width})/2:y=(in_h-{target_height})/2"

        command = [
            "ffmpeg",
            "-i", input_path,
            "-vf", filter_command,
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "medium",
            "-c:a", "copy", # 或者使用 '-an' 禁用音频，或 '-c:a aac -b:a 128k' 重新编码音频
            "-y", # 覆盖输出文件
            output_path
        ]

        print(f"正在执行 ffmpeg 命令: {' '.join(shlex.quote(str(s)) for s in command)}")

        # 执行命令
        # 使用 subprocess.run 等待命令完成
        # check=True 会在 ffmpeg 返回非零退出码（表示错误）时抛出 CalledProcessError
        # capture_output=True 捕获 stdout 和 stderr
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        print("视频裁剪完成！")
        # print("FFmpeg stdout:\n", result.stdout) # 一般没什么用，除非调试
        # print("FFmpeg stderr:\n", result.stderr) # ffmpeg 的进度和信息通常在 stderr 输出

    except FileNotFoundError:
         print("错误：找不到 ffmpeg 命令。请确保 ffmpeg 已安装并在系统 PATH 中。")
    except subprocess.CalledProcessError as e:
        print(f"错误：ffmpeg 执行失败 (返回码 {e.returncode})")
        print(f"ffmpeg 命令: {' '.join(shlex.quote(str(s)) for s in e.cmd)}")
        print(f"ffmpeg stderr:\n{e.stderr}")
    except Exception as e:
        print(f"处理视频时发生未知错误: {e}")

# --- 使用示例 ---
if __name__ == "__main__":
    input_video = "/home/admin123/projects/lerobot-plus/lerobot-joycon_plus_data/datasets/pick_put_hand/images/observation.images.webcam/episode_000000/cam_vis_random_color/aitviewer/video_0.mp4"  # 替换成你的输入视频路径
    output_video = "output_cropped_640x480_ffmpeg.mp4" # 输出视频路径
    crop_width = 640
    crop_height = 480

    # 检查输入文件是否存在（示例中）
    if not os.path.exists(input_video):
         print(f"请将输入视频文件 '{input_video}' 放在脚本同目录下，或修改路径。")
    else:
        crop_video_center_ffmpeg(input_video, output_video, crop_width, crop_height)