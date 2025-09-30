import cv2
import os
import natsort  # For natural sorting of filenames
import argparse

def create_video_from_images(image_folder, video_path, fps=30, sort_numerically=True):
    """
    将指定文件夹中的图片合成为视频。

    Args:
        image_folder (str): 包含图片的文件夹路径。
        video_path (str): 输出视频文件的完整路径 (例如 'output/my_video.mp4')。
        fps (int): 输出视频的帧率 (Frames Per Second)。
        sort_numerically (bool): 是否使用自然排序对图片文件名进行排序。
                                 设为 True 对 'img1.png', 'img10.png', 'img2.png' 这样的文件名排序更好。
                                 设为 False 使用标准字母排序。
    """
    print(f"--- Starting Video Creation ---")
    print(f"Image Source: {image_folder}")
    print(f"Output Video: {video_path}")
    print(f"Target FPS: {fps}")

    # 1. 获取图片文件列表并排序
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    image_files = [f for f in os.listdir(image_folder)
                   if os.path.isfile(os.path.join(image_folder, f)) and f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"Error: No images with extensions {valid_extensions} found in '{image_folder}'")
        return

    # 排序
    if sort_numerically:
        try:
            print("Sorting image files numerically (using natsort)...")
            image_files = natsort.natsorted(image_files)
        except ImportError:
            print("Warning: 'natsort' library not found. Falling back to basic alphabetical sort.")
            print("Install it via 'pip install natsort' for better numerical sorting.")
            image_files.sort()
            sort_numerically = False # Update flag since we fell back
    else:
        print("Sorting image files alphabetically...")
        image_files.sort()

    if not sort_numerically:
         print("Using alphabetical sort. Ensure filenames are padded with zeros (e.g., 001.jpg, 002.jpg) for correct order if needed.")


    image_paths = [os.path.join(image_folder, f) for f in image_files]
    total_images = len(image_paths)
    print(f"Found {total_images} images.")

    # 2. 读取第一张图片获取视频尺寸
    first_image_path = image_paths[0]
    print(f"Reading dimensions from first image: {os.path.basename(first_image_path)}")
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Error: Could not read the first image '{first_image_path}'. Check file format and path.")
        return
    height, width, layers = frame.shape
    size = (width, height)
    print(f"Video dimensions set to: {width}x{height}")

    # 3. 确定 FourCC 编解码器
    #    常见的选择:
    #    - '.mp4': 'mp4v' (兼容性好), 'avc1' (H.264, 压缩率高, 可能需要特定后端), 'h264'
    #    - '.avi': 'XVID' (常用), 'MJPG' (Motion JPEG)
    filename, ext = os.path.splitext(video_path)
    ext = ext.lower()
    if ext == '.mp4':
        # 尝试使用 H.264 (avc1), 如果失败或不可用，可以回退到 'mp4v'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print(f"Using FourCC codec: 'mp4v' for {ext}")
    elif ext == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        print(f"Using FourCC codec: 'XVID' for {ext}")
    else:
        print(f"Warning: Unsupported video extension '{ext}'. Defaulting to '.mp4' with 'mp4v' codec.")
        video_path = filename + '.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print(f"Using FourCC codec: 'mp4v' for .mp4")

    # 4. 初始化 VideoWriter
    # 确保输出目录存在
    output_dir = os.path.dirname(video_path)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    try:
        out = cv2.VideoWriter(video_path, fourcc, fps, size)
        if not out.isOpened():
             raise IOError(f"VideoWriter failed to open. Check codec ('{fourcc}') availability, path ('{video_path}'), and permissions.")
    except Exception as e:
        print(f"Error initializing VideoWriter: {e}")
        print("Consider trying a different codec (e.g., 'mp4v' for .mp4, 'XVID' for .avi) or check OpenCV installation/dependencies (like FFmpeg).")
        return

    # 5. 逐帧写入视频
    print("Processing images and writing video frames...")
    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Skipping corrupted or unreadable image: {os.path.basename(image_path)}")
            continue

        # (可选) 检查并调整尺寸，以防万一有图片尺寸不一致
        h, w, _ = img.shape
        if (w, h) != size:
            print(f"Warning: Image '{os.path.basename(image_path)}' has different dimensions ({w}x{h}). Resizing to {size[0]}x{size[1]}.")
            img = cv2.resize(img, size)

        out.write(img)

        # 打印进度
        if (i + 1) % 20 == 0 or (i + 1) == total_images: # 每 20 帧或最后一帧打印一次
             print(f"  Processed frame {i + 1}/{total_images} ({os.path.basename(image_path)})")


    # 6. 释放资源
    out.release()
    cv2.destroyAllWindows() # 虽然这里没显示窗口，但有时是好习惯

    print("-" * 30)
    print(f"Video successfully created: {video_path}")
    print("-" * 30)

# --- 主程序入口 ---
if __name__ == "__main__":
    # 使用 argparse 使得脚本可以从命令行接收参数，更灵活
    parser = argparse.ArgumentParser(description="Create a video from a sequence of images in a directory.")
    parser.add_argument("image_dir", help="Directory containing the input image files.")
    parser.add_argument("output_video", help="Path for the output video file (e.g., output/result.mp4 or C:/videos/out.avi).")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video (default: 30).")
    parser.add_argument("--sort", choices=['numeric', 'alpha'], default='numeric',
                        help="Sorting method for image files ('numeric' requires natsort, 'alpha' is basic alphabetical sort). Default: numeric.")

    args = parser.parse_args()

    # 简单验证输入目录是否存在
    if not os.path.isdir(args.image_dir):
        print(f"Error: Input directory not found: {args.image_dir}")
        exit(1) # 退出脚本

    use_numeric_sort = (args.sort == 'numeric')

    # 调用核心函数
    create_video_from_images(args.image_dir, args.output_video, args.fps, sort_numerically=use_numeric_sort)