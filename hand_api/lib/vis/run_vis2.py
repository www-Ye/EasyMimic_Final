import os
import cv2
import numpy as np
import torch
import trimesh
import random
# 假设你的 viewer 和 tools 模块路径如下：
import lib.vis.viewer as viewer_utils
from lib.vis.wham_tools.tools import checkerboard_geometry

def camera_marker_geometry(radius, height):
    vertices = np.array(
        [
            [-radius, -radius, 0],
            [radius, -radius, 0],
            [radius, radius, 0],
            [-radius, radius, 0],
            [0, 0, -height],
        ]
    )

    faces = np.array(
        [[0, 1, 2], [0, 2, 3], [1, 0, 4], [2, 1, 4], [3, 2, 4], [0, 3, 4]]
    )

    face_colors = np.array(
        [
            [0.5, 0.5, 0.5, 1.0],
            [0.5, 0.5, 0.5, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ]
    )
    return vertices, faces, face_colors

def run_vis2_on_video_cam_3d(res_dict, res_dict2, output_pth, focal_length, image_names, R_w2c=None, t_w2c=None):
    img0 = cv2.imread(image_names[0])
    img0 = cv2.resize(img0, (1600, 900))
    height, width, _ = img0.shape

    world_mano = {}
    world_mano['vertices'] = res_dict['vertices']
    world_mano['faces'] = res_dict['faces']

    world_mano2 = {}
    world_mano2['vertices'] = res_dict2['vertices']
    world_mano2['faces'] = res_dict2['faces']

    vis_dict = {}
    color_idx = 0
    world_mano['vertices'] = world_mano['vertices']
    for _id, _verts in enumerate(world_mano['vertices']):
        verts = _verts.cpu().numpy()  # T, N, 3
        body_faces = world_mano['faces']
        body_meshes = {
            "v3d": verts,
            "f3d": body_faces,
            "vc": None,
            "name": f"hand_{_id}",
            "color": "director-purple",
        }
        vis_dict[f"hand_{_id}"] = body_meshes
        color_idx += 1

    world_mano2['vertices'] = world_mano2['vertices']
    for _id, _verts in enumerate(world_mano2['vertices']):
        verts = _verts.cpu().numpy()  # T, N, 3
        body_faces = world_mano2['faces']
        body_meshes = {
            "v3d": verts,
            "f3d": body_faces,
            "vc": None,
            "name": f"hand2_{_id}",
            "color": "director-blue",
        }
        vis_dict[f"hand2_{_id}"] = body_meshes
        color_idx += 1

    # 可选：添加相机坐标系原点和轴线可视化
    # 添加坐标系原点和轴线的代码，如果需要的话

    num_frames = len(world_mano['vertices'][_id])
    
    # 关键修改1：创建适合相机坐标系的观察矩阵
    # 在相机坐标系中，我们设置一个固定的视角来观察手部
    # 这个视角应该与相机坐标系对齐
    
    # 方法1：使用固定观察点
    # side_source = torch.tensor([0.0, 0.0, 1.0])  # 从Z轴正方向观察
    # side_target = torch.tensor([0.0, 0.0, 0.0])  # 看向原点
    # up = torch.tensor([0.0, -1.0, 0.0])          # Y轴向上
    # view_camera = lookat_matrix(side_source, side_target, up)
    # viewer_Rt = np.tile(view_camera[:3, :4], (num_frames, 1, 1))
    
    # 方法2：也可以使用单位矩阵作为观察矩阵，直接从相机视角观察
    viewer_Rt = np.zeros((num_frames, 3, 4))
    viewer_Rt[:, :3, :3] = np.eye(3)
    viewer_Rt[:, :3, 3] = np.zeros(3)

    meshes = viewer_utils.construct_viewer_meshes(
        vis_dict, draw_edges=False, flat_shading=False
    )

    vis_h, vis_w = (height, width)
    K = np.array(
        [
            [1000, 0, vis_w / 2],
            [0, 1000, vis_h / 2],
            [0, 0, 1],
        ]
    )
    
    print(vis_w, vis_h)
    assert vis_w == 1600 and vis_h == 900, "分辨率不匹配"
    
    # 关键修改2：使用适合3D查看的相机参数
    # 注意：此处我们不使用图像背景，因为我们是在3D空间中查看手部
    data = viewer_utils.ViewerData(viewer_Rt, K, vis_w, vis_h)
    batch = (meshes, data)

    # 使用 interactive=False 和 render_types=['video']
    breakpoint()
    viewer = viewer_utils.ARCTICViewer(interactive=False, size=(vis_w, vis_h), render_types=['video'])

    # 确保输出目录存在
    os.makedirs(os.path.join(output_pth, 'aitviewer'), exist_ok=True)

    # 删除可能存在的旧视频文件
    video_path = os.path.join(output_pth, 'aitviewer', "cam_3d_video_0.mp4")
    if os.path.exists(video_path):
        os.remove(video_path)

    viewer.render_seq(batch, out_folder=os.path.join(output_pth, 'aitviewer'))
    return video_path

def run_vis2_on_video(res_dict, res_dict2, output_pth, focal_length, image_names, R_c2w=None, t_c2w=None):
    img0 = cv2.imread(image_names[0])
    img0 = cv2.resize(img0, (1600, 900))
    height, width, _ = img0.shape

    world_mano = {}
    world_mano['vertices'] = res_dict['vertices']
    world_mano['faces'] = res_dict['faces']

    world_mano2 = {}
    world_mano2['vertices'] = res_dict2['vertices']
    world_mano2['faces'] = res_dict2['faces']

    vis_dict = {}
    color_idx = 0
    world_mano['vertices'] = world_mano['vertices']
    for _id, _verts in enumerate(world_mano['vertices']):
        verts = _verts.cpu().numpy()  # T, N, 3
        body_faces = world_mano['faces']
        body_meshes = {
            "v3d": verts,
            "f3d": body_faces,
            "vc": None,
            "name": f"hand_{_id}",
            "color": "director-purple",
        }
        vis_dict[f"hand_{_id}"] = body_meshes
        color_idx += 1

    world_mano2['vertices'] = world_mano2['vertices']
    for _id, _verts in enumerate(world_mano2['vertices']):
        verts = _verts.cpu().numpy()  # T, N, 3
        body_faces = world_mano2['faces']
        body_meshes = {
            "v3d": verts,
            "f3d": body_faces,
            "vc": None,
            "name": f"hand2_{_id}",
            "color": "director-blue",
        }
        vis_dict[f"hand2_{_id}"] = body_meshes
        color_idx += 1

    v, f, vc, fc = checkerboard_geometry(length=100, c1=0, c2=0, up="z")
    v[:, 2] -= 2  # z plane
    gound_meshes = {
        "v3d": v,
        "f3d": f,
        "vc": vc,
        "name": "ground",
        "fc": fc,
        "color": -1,
    }
    vis_dict["ground"] = gound_meshes

    num_frames = len(world_mano['vertices'][_id])
    Rt = np.zeros((num_frames, 3, 4))
    Rt[:, :3, :3] = R_c2w[:num_frames]
    Rt[:, :3, 3] = t_c2w[:num_frames]

    verts, faces, face_colors = camera_marker_geometry(0.05, 0.1)
    verts = np.einsum("tij,nj->tni", Rt[:, :3, :3], verts) + Rt[:, None, :3, 3]
    camera_meshes = {
        "v3d": verts,
        "f3d": faces,
        "vc": None,
        "name": "camera",
        "fc": face_colors,
        "color": -1,
    }
    vis_dict["camera"] = camera_meshes

    side_source = torch.tensor([0.463, -0.478, 2.456])
    side_target = torch.tensor([0.026, -0.481, -3.184])
    up = torch.tensor([1.0, 0.0, 0.0])
    view_camera = lookat_matrix(side_source, side_target, up)
    viewer_Rt = np.tile(view_camera[:3, :4], (num_frames, 1, 1))

    meshes = viewer_utils.construct_viewer_meshes(
        vis_dict, draw_edges=False, flat_shading=False
    )

    vis_h, vis_w = (height, width)
    K = np.array(
        [
            [1000, 0, vis_w / 2],
            [0, 1000, vis_h / 2],
            [0, 0, 1],
        ]
    )
    
    print(vis_w, vis_h)
    assert vis_w == 1600 and vis_h == 900, "分辨率不匹配"
    
    data = viewer_utils.ViewerData(viewer_Rt, K, vis_w, vis_h)
    batch = (meshes, data)

    # 使用 interactive=False 和 render_types=['video']
    viewer = viewer_utils.ARCTICViewer(interactive=False, size=(vis_w, vis_h), render_types=['video'])

    # 确保输出目录存在
    os.makedirs(os.path.join(output_pth, 'aitviewer'), exist_ok=True)

    # 删除可能存在的旧视频文件
    video_path = os.path.join(output_pth, 'aitviewer', "world_video_0.mp4")
    if os.path.exists(video_path):
        os.remove(video_path)

    viewer.render_seq(batch, out_folder=os.path.join(output_pth, 'aitviewer'))
    return video_path


def run_vis2_on_video_cam(res_dict, res_dict2, output_pth, focal_length, image_names, R_w2c=None, t_w2c=None, valid_hand=[0,1], v=None):
    img0 = cv2.imread(image_names[0])
    height, width, _ = img0.shape

    world_mano = {}
    world_mano['vertices'] = res_dict['vertices']
    world_mano['faces'] = res_dict['faces']

    world_mano2 = {}
    world_mano2['vertices'] = res_dict2['vertices']
    world_mano2['faces'] = res_dict2['faces']

    vis_dict = {}
    color_idx = 0

    num_frames = None
    if 0 in valid_hand:
        world_mano['vertices'] = world_mano['vertices']
        for _id, _verts in enumerate(world_mano['vertices']):
            verts = _verts.cpu().numpy()  # T, N, 3
            body_faces = world_mano['faces']
            body_meshes = {
                "v3d": verts,
                "f3d": body_faces,
                "vc": None,
                "name": f"hand_{_id}",
                # "color": "director-purple",
                "color": "random",
            }
            vis_dict[f"hand_{_id}"] = body_meshes
            color_idx += 1

        num_frames = len(world_mano['vertices'][_id])

    if 1 in valid_hand:
        world_mano2['vertices'] = world_mano2['vertices']
        for _id, _verts in enumerate(world_mano2['vertices']):
            verts = _verts.cpu().numpy()  # T, N, 3
            body_faces = world_mano2['faces']
            body_meshes = {
                "v3d": verts,
                "f3d": body_faces,
                "vc": None,
                "name": f"hand2_{_id}",
                # "color": "director-blue",
                "color": "random",
            }
            vis_dict[f"hand2_{_id}"] = body_meshes
            color_idx += 1

        if num_frames is None:
            num_frames = len(world_mano2['vertices'][_id])

    meshes = viewer_utils.construct_viewer_meshes(
        vis_dict, draw_edges=False, flat_shading=False
    )

    Rt = np.zeros((num_frames, 3, 4))
    Rt[:, :3, :3] = R_w2c[:num_frames]
    Rt[:, :3, 3] = t_w2c[:num_frames]

    cols, rows = (width, height)
    K = np.array(
        [
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1],
        ]
    )
    vis_h = height
    vis_w = width

    data = viewer_utils.ViewerData(Rt, K, cols, rows, imgnames=image_names)
    batch = (meshes, data)

    # 使用 interactive=False 和 render_types=['video']
    viewer = viewer_utils.ARCTICViewer(interactive=False, size=(vis_w, vis_h), render_types=['video'], v=v)

    # 确保输出目录存在
    os.makedirs(os.path.join(output_pth, 'aitviewer'), exist_ok=True)

    # 删除可能存在的旧视频文件
    video_path = os.path.join(output_pth, 'aitviewer', "video_0.mp4")
    if os.path.exists(video_path):
        os.remove(video_path)

    viewer.render_seq(batch, out_folder=os.path.join(output_pth, 'aitviewer'))
    
    return video_path

def run_vis2_on_video_cam_random_color(res_dict, res_dict2, output_pth, focal_length, 
                                       image_names, R_w2c=None, t_w2c=None, valid_hand=[0,1], 
                                       v=None, video_name=None, enhanced_finger_indices=None):
    """
    增强版渲染函数，支持指定手部节点进行颜色增强
    
    Args:
        enhanced_finger_indices: list or None, 要增强的手部节点索引列表
                                例如 [4, 8] 表示只增强拇指尖(4)和食指尖(8)   
                                如果为 None，则增强整个手部模型
    """
    img0 = cv2.imread(image_names[0])
    height, width, _ = img0.shape

    world_mano = {}
    if res_dict is not None:
        world_mano['vertices'] = res_dict['vertices']
        world_mano['faces'] = res_dict['faces']

    world_mano2 = {}    
    if res_dict2 is not None:
        world_mano2['vertices'] = res_dict2['vertices']
        world_mano2['faces'] = res_dict2['faces']

    vis_dict = {}
    color_idx = 0

    num_frames = None

    no_overlay_prob = 0.2

    if 0 in valid_hand:
        world_mano['vertices'] = world_mano['vertices']
        for _id, _verts in enumerate(world_mano['vertices']):
            verts = _verts.cpu().numpy()
            body_faces = world_mano['faces']
            num_frames_hand = verts.shape[0]
            num_verts_hand = verts.shape[1]

            # TODO 修改为部分mesh增强！！
            # --- 生成每帧随机颜色或透明 ---
            vertex_colors_per_frame = np.zeros((num_frames_hand, num_verts_hand, 4)) # 初始化为 RGBA
            for f in range(num_frames_hand):
                # 增强整个手部模型
                if random.random() < no_overlay_prob:
                    # 设为透明 (Alpha=0)
                    vertex_colors_per_frame[f, :, :] = [0.0, 0.0, 0.0, 0.0]
                else:
                    # 生成随机 RGB
                    rand_color_rgb = np.random.rand(1, 3)
                    # 组合成 RGBA (Alpha=1)
                    rand_color_rgba = np.concatenate([rand_color_rgb, [[1.0]]], axis=1)
                    # 应用到该帧所有顶点
                    vertex_colors_per_frame[f, :, :] = np.tile(rand_color_rgba, (num_verts_hand, 1))
               
            # --- 颜色生成结束 ---

            body_meshes = {
                "v3d": verts,
                "f3d": body_faces,
                "vc": vertex_colors_per_frame,
                "name": f"hand_{_id}",
            }
            vis_dict[f"hand_{_id}"] = body_meshes

            if num_frames is None:
                num_frames = num_frames_hand

    if 1 in valid_hand:
        world_mano2['vertices'] = world_mano2['vertices']
        for _id, _verts in enumerate(world_mano2['vertices']):
            verts = _verts.cpu().numpy()
            body_faces = world_mano2['faces']
            num_frames_hand2 = verts.shape[0]
            num_verts_hand2 = verts.shape[1]

            # --- 生成每帧随机颜色或透明 ---
            vertex_colors_per_frame2 = np.zeros((num_frames_hand2, num_verts_hand2, 4))
            for f in range(num_frames_hand2):
                # 增强整个手部模型
                if random.random() < no_overlay_prob: # 独立为第二只手决定是否透明
                    vertex_colors_per_frame2[f, :, :] = [0.0, 0.0, 0.0, 0.0]
                else:
                    rand_color_rgb2 = np.random.rand(1, 3)
                    rand_color_rgba2 = np.concatenate([rand_color_rgb2, [[1.0]]], axis=1)
                    vertex_colors_per_frame2[f, :, :] = np.tile(rand_color_rgba2, (num_verts_hand2, 1))
            
            # --- 颜色生成结束 ---

            body_meshes = {
                "v3d": verts,
                "f3d": body_faces,
                "vc": vertex_colors_per_frame2,
                "name": f"hand2_{_id}",
            }
            vis_dict[f"hand2_{_id}"] = body_meshes

            if num_frames is None:
                num_frames = num_frames_hand2
            elif num_frames != num_frames_hand2:
                 print(f"Warning: Frame count mismatch ({num_frames} vs {num_frames_hand2}). Using minimum.")
                 num_frames = min(num_frames, num_frames_hand2)

    meshes = viewer_utils.construct_viewer_meshes_random_color(
        vis_dict, draw_edges=False, flat_shading=False
    )

    Rt = np.zeros((num_frames, 3, 4))
    Rt[:, :3, :3] = R_w2c[:num_frames]
    Rt[:, :3, 3] = t_w2c[:num_frames]

    cols, rows = (width, height)
    K = np.array(
        [
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1],
        ]
    )
    vis_h = height
    vis_w = width

    print("width, height", width, height)
    print("cols, rows", cols, rows)
    print("vis_w, vis_h", vis_w, vis_h)
    # 640 480
    # breakpoint()

    data = viewer_utils.ViewerData(Rt, K, cols, rows, imgnames=image_names)
    batch = (meshes, data)

    # breakpoint()

    # 使用 interactive=False 和 render_types=['video']
    viewer = viewer_utils.ARCTICViewer(interactive=False, size=(vis_w, vis_h), render_types=['video'], v=v)

    # breakpoint()

    # 确保输出目录存在
    os.makedirs(output_pth, exist_ok=True)

    # 删除可能存在的旧视频文件
    video_path = os.path.join(output_pth, f"{video_name}.mp4")
    print("video_path", video_path)
    if os.path.exists(video_path):
        os.remove(video_path)

    viewer.render_seq(batch, out_folder=output_pth, video_name=video_name)

    # breakpoint()
    
    return video_path

def lookat_matrix(source_pos, target_pos, up):
    """
    IMPORTANT: USES RIGHT UP BACK XYZ CONVENTION
    :param source_pos (*, 3)
    :param target_pos (*, 3)
    :param up (3,)
    """
    *dims, _ = source_pos.shape
    up = up.reshape(*(1,) * len(dims), 3)
    up = up / (torch.linalg.norm(up, dim=-1, keepdim=True) + 1e-8)  # 防止除以零
    back = normalize(target_pos - source_pos)
    right = normalize(torch.linalg.cross(up, back))
    up = normalize(torch.linalg.cross(back, right))
    R = torch.stack([right, up, back], dim=-1)
    return make_4x4_pose(R, source_pos)

def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)

def normalize(x):
    return x / (torch.linalg.norm(x, dim=-1, keepdim=True) + 1e-8)  # 防止除以零

def save_mesh_to_obj(vertices, faces, file_path):
    # 创建一个 Trimesh 对象
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # 导出为 .obj 文件
    mesh.export(file_path)
    print(f"Mesh saved to {file_path}")


# 使用PyRender进行手部可视化并保存视频
def get_predefined_finger_regions():
    """
    基于MANO模型拓扑的手指区域,使用Mano lbs_weight 计算
    注意：这些范围是近似的，可能需要根据具体应用调整
    """
    # finger_regions = {
    #     'thumb': list(range(745-20, 778)),      # 大拇指区域
    #     'index': list(range(317-20, 350+20)),      # 食指区域  
    #     'middle': list(range(440, 490)),     # 中指区域
    #     'ring': list(range(550, 600)),       # 无名指区域
    #     'pinky': list(range(670, 720)),      # 小拇指区域
    #     'palm': list(range(0, 100))          # 手掌区域
    # }

    new_finger_regions = {
        'thumb': [699, 700, 701, 703, 704, 705, 702, 706, 124, 125, 249, 698, 697, 250, 251, 287, 267,  31, 754,  28, 104, 714, 753, 711, 710, 713, 708, 252, 248, 253, 712, 123, 126, 709, 707, 286,  29,  30, 266,  10, 105, 755,  43, 741, 757, 742,  89, 758, 236,   6,
                  717, 719, 720, 722, 723, 724, 725, 726, 727, 728, 729, 730, 744, 746, 747, 748, 751, 752, 740, 718, 749, 745, 735, 736, 765, 750, 739, 734, 733, 737, 716, 763, 738, 743, 766, 760, 759, 756, 721, 764, 761, 768, 762, 767,  71, 731, 732, 758, 742, 757,
                  744],    # 大拇指区域
        'index': [177, 136, 135, 173, 140, 261, 139, 176, 171, 170, 194, 164, 174, 165, 260, 133, 195, 175, 212, 189, 138, 168, 134, 258, 274, 172, 263, 137, 169, 166,  87,  48, 167,  59, 225,  49, 186, 213, 132, 156, 259,  58,  62, 150, 128, 430, 457, 414, 415, 420,
                  245, 237, 283, 273, 238, 222, 281, 282, 224, 221, 272,  46, 223, 155, 226,  47, 280,  57,  56,  86, 340, 298, 295, 341, 297, 301,  58, 299, 213, 156,  49, 225, 294,  59, 167, 300,  87, 296,  48, 166, 330, 134, 331, 342, 344, 457, 430, 431, 456, 441,
                  320]     # 食指区域  
    }
    return new_finger_regions

def get_predefined_finger_joints():
    """
    基于MANO模型拓扑的手指关节,使用Mano lbs_weight 计算
    注意：这些范围是近似的，可能需要根据具体应用调整
    """
    new_finger_joints = {
        'thumb_pip':699,
        'thumb_dip':717,
        'thumb_tip':744,
        'index_pip':177,
        'index_dip':245,
        'index_tip':320,
    }
    return new_finger_joints
        
def get_finger_faces_predefined(faces, finger_name):
    """
    使用预定义区域获取手指faces
    """
    finger_regions = get_predefined_finger_regions()
    if finger_name not in finger_regions:
        raise ValueError(f"Unknown finger: {finger_name}")
    
    finger_vertices = finger_regions[finger_name]
    
    # 找到包含这些顶点的面片
    face_mask = torch.isin(faces, torch.tensor(finger_vertices)).any(dim=1)
    finger_faces = faces[face_mask]
    
    return finger_faces

def get_multiple_finger_faces(faces, finger_names):
    """
    获取多个手指的faces
    
    Args:
        faces: [F, 3] 原始面片索引
        finger_names: list of str, 要显示的手指名称
    
    Returns:
        combined_faces: 合并后的面片
        finger_face_mapping: 每个手指对应的面片映射
    """
    all_finger_faces = []
    finger_face_mapping = {}
    
    for finger_name in finger_names:
        finger_faces = get_finger_faces_predefined(faces, finger_name)
        finger_face_mapping[finger_name] = finger_faces
        all_finger_faces.append(finger_faces)
    
    # 合并所有手指的faces
    if all_finger_faces:
        combined_faces = torch.cat(all_finger_faces, dim=0)
        # 去除重复的faces
        combined_faces = torch.unique(combined_faces, dim=0)
    else:
        combined_faces = torch.empty((0, 3), dtype=faces.dtype)
    
    return combined_faces, finger_face_mapping

def visualize_custom_fingers(res_dict, valid_hand, finger_names):
    """
    可视化指定的手指
    
    Args:
        res_dict: 包含vertices和faces的字典
        valid_hand: 有效手部ID列表
        finger_names: 要可视化的手指名称列表，如 ['thumb', 'index', 'middle']
    
    Returns:
        hands_data: 处理后的手部数据
    """
    hands_data = {}
    
    if res_dict is not None and 0 in valid_hand:
        for _id, _verts in enumerate(res_dict['vertices']):
            verts = _verts.cpu().numpy() if hasattr(_verts, 'cpu') else _verts
            all_faces = res_dict['faces']
            
            if not isinstance(all_faces, torch.Tensor):
                all_faces = torch.tensor(all_faces)
            
            # 获取指定手指的faces
            combined_faces, finger_mapping = get_multiple_finger_faces(all_faces, finger_names)
            
            print(f"Hand {_id} - Visualizing fingers: {finger_names}")
            print(f"  Original faces: {all_faces.shape[0]}")
            for finger_name in finger_names:
                print(f"  {finger_name.capitalize()} faces: {finger_mapping[finger_name].shape[0]}")
            print(f"  Combined faces: {combined_faces.shape[0]}")
            
            hands_data[f'hand_{_id}'] = {
                'vertices': verts,
                'faces': combined_faces.numpy(),
                'finger_mapping': {k: v.numpy() for k, v in finger_mapping.items()}
            }

    if res_dict is not None and 1 in valid_hand:
        for _id, _verts in enumerate(res_dict['vertices']):
            verts = _verts.cpu().numpy() if hasattr(_verts, 'cpu') else _verts
            all_faces = res_dict['faces']
            
            if not isinstance(all_faces, torch.Tensor):
                all_faces = torch.tensor(all_faces)
            
            # 获取指定手指的faces
            combined_faces, finger_mapping = get_multiple_finger_faces(all_faces, finger_names)
            
            print(f"Hand {_id} - Visualizing fingers: {finger_names}")
            print(f"  Original faces: {all_faces.shape[0]}")
            for finger_name in finger_names:
                print(f"  {finger_name.capitalize()} faces: {finger_mapping[finger_name].shape[0]}")
            print(f"  Combined faces: {combined_faces.shape[0]}")
            
            hands_data[f'hand2_{_id}'] = {
                'vertices': verts,
                'faces': combined_faces.numpy(),
                'finger_mapping': {k: v.numpy() for k, v in finger_mapping.items()}
            }
            

    
    return hands_data

def pyrender_on_video_cam_random_color(res_dict, res_dict2, output_pth, focal_length, 
                                       image_names, R_w2c=None, t_w2c=None, valid_hand=[0,1], 
                                       v=None, video_name=None, enhanced_finger_indices=None,joints=None,
                                       data_augmentation_type="none",flicker_interval=5, fast_render=False):
    """
    使用PyRender进行手部可视化并保存视频

    Args:
        res_dict: dict, 第一个手的MANO输出 {'vertices': list, 'faces': array}
        res_dict2: dict, 第二个手部数据 {'vertices': list, 'faces': array}
        output_pth: str, 输出路径
        focal_length: float, 相机焦距
        image_names: list, 背景图像路径列表
        R_w2c: array, 世界到相机的旋转矩阵
        t_w2c: array, 世界到相机的平移向量
        valid_hand: list, 有效的手部索引
        v: 详细程度
        video_name: str, 视频文件名
        enhanced_finger_indices: list, 增强的手指索引
        joints: array,  [T, 21, 3] 关节位置
        data_augmentation_type: str, 数据增强类型
        flicker_interval: int, 闪烁间隔
    """
    import pyrender
    import imageio
    from tqdm import trange

    # 读取第一帧图像获取尺寸
    img0 = cv2.imread(image_names[0])
    height, width, _ = img0.shape

    # 准备手部数据
    hands_data = {}

    if data_augmentation_type == "none":
        # 处理第一个手部数据
        if res_dict is not None:
            hands_data_1 = {}
            for _id, _verts in enumerate(res_dict['vertices']):
                verts = _verts.cpu().numpy() if hasattr(_verts, 'cpu') else _verts
                faces = res_dict['faces']
                if not isinstance(faces, np.ndarray):
                    faces = faces.cpu().numpy() if hasattr(faces, 'cpu') else np.array(faces)
                hands_data_1[f'hand_{_id}'] = {
                    'vertices': verts,
                    'faces': faces,
                }
            hands_data.update(hands_data_1)
        # 处理第二个手部数据
        if res_dict2 is not None:
            hands_data_2 = {}
            for _id, _verts in enumerate(res_dict2['vertices']):
                verts = _verts.cpu().numpy() if hasattr(_verts, 'cpu') else _verts
                faces = res_dict2['faces']
                if not isinstance(faces, np.ndarray):
                    faces = faces.cpu().numpy() if hasattr(faces, 'cpu') else np.array(faces)
                hands_data_2[f'hand2_{_id}'] = {
                    'vertices': verts,
                    'faces': faces,
                }
            hands_data.update(hands_data_2)
    else:
        # 处理第一个手部数据
        if res_dict is not None:
            if data_augmentation_type == "full_hand":
                # 结构与visualize_custom_fingers一致，保证hand_0/hand_1等key
                hands_data_1 = {}
                for _id, _verts in enumerate(res_dict['vertices']):
                    verts = _verts.cpu().numpy() if hasattr(_verts, 'cpu') else _verts
                    faces = res_dict['faces']
                    if not isinstance(faces, np.ndarray):
                        faces = faces.cpu().numpy() if hasattr(faces, 'cpu') else np.array(faces)
                    hands_data_1[f'hand_{_id}'] = {
                        'vertices': verts,
                        'faces': faces,
                    }
            elif data_augmentation_type in ["partial_hand", "partial_hand_joint"]:
                hands_data_1 = visualize_custom_fingers(res_dict, valid_hand, ['thumb', 'index'])
            else:
                raise ValueError(f"Invalid data augmentation type: {data_augmentation_type}")
            hands_data.update(hands_data_1)

        # 处理第二个手部数据
        if res_dict2 is not None:
            if data_augmentation_type == "full_hand":
                hands_data_2 = {}
                for _id, _verts in enumerate(res_dict2['vertices']):
                    verts = _verts.cpu().numpy() if hasattr(_verts, 'cpu') else _verts
                    faces = res_dict2['faces']
                    if not isinstance(faces, np.ndarray):
                        faces = faces.cpu().numpy() if hasattr(faces, 'cpu') else np.array(faces)
                    hands_data_2[f'hand2_{_id}'] = {
                        'vertices': verts,
                        'faces': faces,
                    }
            elif data_augmentation_type in ["partial_hand", "partial_hand_joint"]:
                hands_data_2 = visualize_custom_fingers(res_dict2, valid_hand, ['thumb', 'index'])
            else:
                raise ValueError(f"Invalid data augmentation type: {data_augmentation_type}")
            hands_data.update(hands_data_2)

    if not hands_data:
        print("No valid hand data found!")
        return None
    

    #########part2#########

    # 获取最大帧数
    max_frames = max(hand_data['vertices'].shape[0] for hand_data in hands_data.values())

    # 设置相机内参
    K = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ])
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 默认网格颜色
    mesh_colors = {
        'hand_0': (1.0, 0.8, 0.7),    # 右手皮肤色
        'hand_1': (0.9, 0.7, 0.6),    # 左手稍微不同的颜色
        'hand2_0': (0.8, 0.9, 0.7),   # 第二个手部数据的颜色
        'hand2_1': (0.7, 0.8, 0.9)
    }

    background_color = (0.0, 0.0, 0.0)
    blend_alpha = 0.8

    print(f"Creating PyRender video with {max_frames} frames...")
    if fast_render:
        print("Fast render mode enabled: using lower quality settings")

    frames = []
    color_cache = {}
    
    # 快速渲染模式下降低分辨率
    render_width = width // 2 if fast_render else width
    render_height = height // 2 if fast_render else height

    for frame_idx in trange(max_frames, desc="PyRender rendering"):
        # 初始化渲染器
        renderer = pyrender.OffscreenRenderer(
            viewport_width=render_width, 
            viewport_height=render_height,
            point_size=1.0
        )

        # 创建场景
        scene = pyrender.Scene(
            bg_color=[*background_color, 1.0],
            ambient_light=(0.3, 0.3, 0.3)
        )

        if data_augmentation_type != "none":
            # 为每个手部添加网格
            for hand_name, hand_data in hands_data.items():
                if frame_idx >= hand_data['vertices'].shape[0]:
                    continue

                # 获取当前帧的顶点和面
                vertices = hand_data['vertices'][frame_idx]  # (V, 3)
                faces = hand_data['faces']  # (F, 3)
                num_verts_hand = vertices.shape[0]
                # --- 生成随机颜色或透明，并可控制RGB闪烁频率 ---

                if hand_name not in color_cache:
                    color_cache[hand_name] = {}

                # 控制RGB闪烁频率
                if (frame_idx % flicker_interval == 0) or ('last_color' not in color_cache[hand_name]):
                    rand_color_rgb = np.random.rand(3)
                    color_cache[hand_name]['last_color'] = rand_color_rgb
                    color_cache[hand_name]['last_frame'] = frame_idx
                else:
                    rand_color_rgb = color_cache[hand_name]['last_color']
                mesh_color = tuple(rand_color_rgb)

                # 创建trimesh
                vertex_colors = np.array([(*mesh_color, 1.0)] * num_verts_hand)

                tri_mesh = trimesh.Trimesh(
                    vertices=vertices,
                    faces=faces,
                    vertex_colors=vertex_colors
                )

                # 应用坐标系变换 (PyRender使用不同的约定)
                transform = np.eye(4)
                transform[1, 1] = -1  # 翻转Y轴
                transform[2, 2] = -1  # 翻转Z轴
                tri_mesh.apply_transform(transform)

                # 创建PyRender网格材质
                material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.1,
                    roughnessFactor=0.8,
                    alphaMode='OPAQUE',
                    baseColorFactor=(*mesh_color, 1.0)
                )

                py_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=material)
                scene.add(py_mesh, name=hand_name)

                # 仅在增强类型为partial_hand_joint时可视化关节和连线
                if data_augmentation_type == "partial_hand_joint":

                    # 只可视化食指和拇指关节
                    vis_idx = [4,8]
                    thumb_pip = joints[frame_idx, 2].cpu().numpy()
                    thumb_tip = joints[frame_idx, 4].cpu().numpy()
                    index_mcp = joints[frame_idx, 5].cpu().numpy()
                    index_tip = joints[frame_idx, 8].cpu().numpy()
                    midpoint = (thumb_pip + index_mcp) / 2
                    midpoint = midpoint.squeeze(0) if midpoint.ndim > 1 else midpoint

                    for joint_idx in vis_idx:
                        joint_pos = joints[frame_idx, joint_idx].cpu().numpy()

                        # 应用相同的坐标变换
                        joint_pos_transformed = transform[:3, :3] @ joint_pos + transform[:3, 3]

                        # 创建球体网格
                        sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.01)
                        sphere.apply_translation(joint_pos_transformed)

                        # 创建黑色材质
                        sphere_material = pyrender.MetallicRoughnessMaterial(
                            metallicFactor=0.0,
                            roughnessFactor=1.0,
                            alphaMode='OPAQUE',
                            baseColorFactor=(0.0, 0.0, 0.0, 1.0)  # 黑色
                        )

                        sphere_mesh = pyrender.Mesh.from_trimesh(sphere, material=sphere_material)
                        scene.add(sphere_mesh, name=f"{hand_name}_joint_{joint_idx}")

                    # 可视化midpoint
                    midpoint_transformed = transform[:3, :3] @ midpoint + transform[:3, 3]

                    # 创建midpoint球体网格
                    midpoint_sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.015)
                    midpoint_sphere.apply_translation(midpoint_transformed)

                    # 创建红色材质
                    midpoint_material = pyrender.MetallicRoughnessMaterial(
                        metallicFactor=0.0,
                        roughnessFactor=1.0,
                        alphaMode='OPAQUE',
                        baseColorFactor=(1.0, 0.0, 0.0, 1.0)  # 红色
                    )

                    midpoint_mesh = pyrender.Mesh.from_trimesh(midpoint_sphere, material=midpoint_material)
                    scene.add(midpoint_mesh, name=f"{hand_name}_midpoint")

                    # 可视化从指尖到midpoint的连线
                    # 拇指指尖连线
                    thumb_tip_transformed = transform[:3, :3] @ thumb_tip + transform[:3, 3]

                    # 计算拇指连线的方向向量和长度
                    thumb_direction = midpoint_transformed - thumb_tip_transformed
                    thumb_length = np.linalg.norm(thumb_direction)

                    # 检查thumb_length是否有效
                    if thumb_length > 1e-6:
                        # 创建圆柱体
                        thumb_cylinder = trimesh.creation.cylinder(radius=0.002, height=thumb_length)

                        # 计算旋转矩阵以对齐圆柱体
                        thumb_direction_norm = thumb_direction / thumb_length
                        # 默认圆柱体沿z轴
                        z_axis = np.array([0, 0, 1])
                        # 计算旋转轴
                        rotation_axis = np.cross(z_axis, thumb_direction_norm)
                        rotation_angle = np.arccos(np.clip(np.dot(z_axis, thumb_direction_norm), -1.0, 1.0))

                        if np.linalg.norm(rotation_axis) > 1e-6:
                            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                            # 创建旋转矩阵
                            cos_angle = np.cos(rotation_angle)
                            sin_angle = np.sin(rotation_angle)
                            K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                                        [rotation_axis[2], 0, -rotation_axis[0]],
                                        [-rotation_axis[1], rotation_axis[0], 0]])
                            rotation_matrix = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
                            thumb_cylinder.apply_transform(np.vstack([np.hstack([rotation_matrix, np.zeros((3, 1))]), [0, 0, 0, 1]]))

                        # 移动到中点位置
                        center_position = (thumb_tip_transformed + midpoint_transformed) / 2
                        thumb_cylinder.apply_translation(center_position)

                        # 创建绿色材质
                        line_material = pyrender.MetallicRoughnessMaterial(
                            metallicFactor=0.0,
                            roughnessFactor=1.0,
                            alphaMode='OPAQUE',
                            baseColorFactor=(0.0, 1.0, 0.0, 1.0)  # 绿色
                        )

                        thumb_line_mesh = pyrender.Mesh.from_trimesh(thumb_cylinder, material=line_material)
                        scene.add(thumb_line_mesh, name=f"{hand_name}_thumb_line")

                    # 食指指尖连线
                    index_tip_transformed = transform[:3, :3] @ index_tip + transform[:3, 3]

                    # 计算食指连线的方向向量和长度
                    index_direction = midpoint_transformed - index_tip_transformed
                    index_length = np.linalg.norm(index_direction)

                    # 检查index_length是否有效
                    if index_length > 1e-6:
                        # 创建圆柱体
                        index_cylinder = trimesh.creation.cylinder(radius=0.002, height=index_length)

                        # 计算旋转矩阵以对齐圆柱体
                        index_direction_norm = index_direction / index_length
                        # 默认圆柱体沿z轴
                        z_axis = np.array([0, 0, 1])
                        # 计算旋转轴
                        rotation_axis = np.cross(z_axis, index_direction_norm)
                        rotation_angle = np.arccos(np.clip(np.dot(z_axis, index_direction_norm), -1.0, 1.0))

                        if np.linalg.norm(rotation_axis) > 1e-6:
                            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                            # 创建旋转矩阵
                            cos_angle = np.cos(rotation_angle)
                            sin_angle = np.sin(rotation_angle)
                            K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                                        [rotation_axis[2], 0, -rotation_axis[0]],
                                        [-rotation_axis[1], rotation_axis[0], 0]])
                            rotation_matrix = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
                            index_cylinder.apply_transform(np.vstack([np.hstack([rotation_matrix, np.zeros((3, 1))]), [0, 0, 0, 1]]))

                        # 移动到中点位置
                        center_position = (index_tip_transformed + midpoint_transformed) / 2
                        index_cylinder.apply_translation(center_position)

                        # 创建绿色材质（如果之前没有创建）
                        if 'line_material' not in locals():
                            line_material = pyrender.MetallicRoughnessMaterial(
                                metallicFactor=0.0,
                                roughnessFactor=1.0,
                                alphaMode='OPAQUE',
                                baseColorFactor=(0.0, 1.0, 0.0, 1.0)  # 绿色
                            )

                        index_line_mesh = pyrender.Mesh.from_trimesh(index_cylinder, material=line_material)
                        scene.add(index_line_mesh, name=f"{hand_name}_index_line")

        # 设置相机
        camera = pyrender.IntrinsicsCamera(
            fx=fx, fy=fy, cx=cx, cy=cy, znear=0.1, zfar=10.0
        )

        camera_pose = np.eye(4)
        scene.add(camera, pose=camera_pose)

        # 添加光照
        # 创建Raymond光照
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            light_node = pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                matrix=matrix
            )
            scene.add_node(light_node)

        # 在相机位置添加点光源
        point_light = pyrender.PointLight(color=np.ones(3), intensity=2.0)
        light_pose = np.eye(4)
        light_pose[2, 3] = 1.0  # 将光源稍微向前移动
        scene.add(point_light, pose=light_pose)

        # 渲染
        try:
            color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            color = color.astype(np.float32) / 255.0
        except Exception as e:
            print(f"Rendering error at frame {frame_idx}: {e}")
            # 出错时创建空白帧
            color = np.zeros((height, width, 4), dtype=np.float32)
            color[:, :, 3] = 1.0  # 完全不透明

        renderer.delete()

        # 处理渲染帧
        if frame_idx < len(image_names):
            # 与背景帧混合
            background = cv2.imread(image_names[frame_idx]).astype(np.float32) / 255.0

            # 如果需要调整背景尺寸
            if background.shape[:2] != (height, width):
                background = cv2.resize(
                    background, (width, height), 
                    interpolation=cv2.INTER_LINEAR
                )

            # 转换BGR到RGB
            background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

            # 快速渲染模式下，需要将渲染结果放大到原始尺寸
            if fast_render:
                color = cv2.resize(color, (width, height), interpolation=cv2.INTER_LINEAR)
                depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)

            mask = depth > 0
            if np.any(mask):
                hand_color = color[mask]
                background[mask] = (hand_color[:, :3] * hand_color[:, 3:4] * blend_alpha + 
                                  background[mask] * (1 - hand_color[:, 3:4] * blend_alpha))
            final_frame = (background * 255).astype(np.uint8)
        else:
            # 快速渲染模式下，需要将渲染结果放大到原始尺寸
            if fast_render:
                color = cv2.resize(color, (width, height), interpolation=cv2.INTER_LINEAR)
            final_frame = (color[:, :, :3] * 255).astype(np.uint8)

        frames.append(final_frame)

    # 保存视频
    os.makedirs(output_pth, exist_ok=True)
    if video_name is None:
        video_name = "pyrender_output.mp4"

    # 删除可能存在的旧视频文件
    video_path = os.path.join(output_pth, f"{video_name}.mp4")
    print("video_path", video_path)
    if os.path.exists(video_path):
        os.remove(video_path)

    print(f"Saving PyRender video to {video_path}...")
    imageio.mimsave(video_path, frames, fps=30, codec='libx264')
    print("PyRender video saved successfully!")

    return video_path


def computer_thumb_index_midpoint(joints):
    """
    计算食指和拇指中点
    """
    if len(joints.shape) == 3:  # [time_steps, 21, 3]
        joints = joints.unsqueeze(0)  # 添加批次维度 [1, time_steps, 21, 3]
    
    batch_size, time_steps, num_joints, _ = joints.shape

    thumb_pip_idx = 2   # 大拇指近指间关节索引
    index_mcp_idx = 5   # 食指掌指关节索引
    thumb_tip = joints[:, :, thumb_pip_idx]   # [batch_size, time_steps, 3]
    index_tip = joints[:, :, index_mcp_idx]   # [batch_size, time_steps, 3] 
    # 计算食指和拇指中点
    midpoint = (thumb_tip + index_tip) / 2
    midpoint = midpoint.squeeze(0)
    print(f"midpoint.shape:{midpoint.shape}")
    return midpoint

# 确保你的 lib 目录在 Python 路径中 (如果不在当前目录)
# import sys
# sys.path.append('/path/to/your/lib')

# 示例用法 (你需要提供实际的输入数据)
if __name__ == '__main__':
    # 假设你有一些模拟数据
    num_frames = 10
    num_verts = 778  # 示例顶点数
    res_dict = {
        'vertices': [torch.randn(num_frames, num_verts, 3) for _ in range(2)],  # 两个手的顶点
        'faces': np.random.randint(0, num_verts, (num_verts * 2, 3)),  # 示例面
    }
    res_dict2 = {
        'vertices': [torch.randn(num_frames, num_verts, 3) for _ in range(2)],
        'faces': np.random.randint(0, num_verts, (num_verts * 2, 3)),
    }
    output_pth = 'output'  # 输出目录
    focal_length = 1000.0
    image_names = ['image.jpg'] * num_frames # 示例图像名列表, 实际中你需要提供真实的图像路径
    #  创建虚拟图像
    if not os.path.exists('image.jpg'):
      img = np.zeros((480, 640, 3), dtype=np.uint8)
      cv2.imwrite('image.jpg', img)

    R_c2w = np.eye(3)  # 示例相机旋转
    t_c2w = np.array([0, 0, 2])   # 示例相机平移

    # 调用渲染函数
    video_file = run_vis2_on_video(res_dict, res_dict2, output_pth, focal_length, image_names, R_c2w=R_c2w, t_c2w=t_c2w)
    print(f"Video saved to: {video_file}")
    #或者
    # video_file = run_vis2_on_video_cam(res_dict, res_dict2, output_pth, focal_length, image_names, R_w2c=R_c2w, t_w2c=t_c2w)
    # print(f"Video saved to: {video_file}")