import os
import os.path as op
import re
from abc import abstractmethod

import matplotlib.cm as cm
import numpy as np
from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.billboard import Billboard
from aitviewer.renderables.meshes import Meshes
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.scene.material import Material
from aitviewer.utils.so3 import aa2rot_numpy
from aitviewer.viewer import Viewer
from easydict import EasyDict as edict
from loguru import logger
from PIL import Image
from tqdm import tqdm
import random

from aitviewer.configuration import CONFIG as C
C.update_conf({"window_type": "pyglet"})

OBJ_ID = 100
SMPLX_ID = 150
LEFT_ID = 200
RIGHT_ID = 250
SEGM_IDS = {"object": OBJ_ID, "smplx": SMPLX_ID, "left": LEFT_ID, "right": RIGHT_ID}

cmap = cm.get_cmap("plasma")
materials = {
    "white": Material(color=(1.0, 1.0, 1.0, 1.0), ambient=0.2),
    "green": Material(color=(0.0, 1.0, 0.0, 1.0), ambient=0.2),
    "blue": Material(color=(0.0, 0.0, 1.0, 1.0), ambient=0.2),
    "red": Material(color=(0.969, 0.106, 0.059, 1.0), ambient=0.2),
    "cyan": Material(color=(0.051, 0.659, 0.051, 1.0), ambient=0.2),
    "light-blue": Material(color=(0.588, 0.5647, 0.9725, 1.0), ambient=0.2),
    "cyan-light": Material(color=(0.051, 0.659, 0.051, 1.0), ambient=0.2),
    "dark-light": Material(color=(0.404, 0.278, 0.278, 1.0), ambient=0.2),
    "rice": Material(color=(0.922, 0.922, 0.102, 1.0), ambient=0.2),
    "whac-whac": Material(color=(167/255, 193/255, 203/255, 1.0), ambient=0.2),
    "whac-wham": Material(color=(165/255, 153/255, 174/255, 1.0), ambient=0.2),
    "pace-blue": Material(color=(0.584, 0.902, 0.976, 1.0), ambient=0.2),
    "pace-green": Material(color=(0.631, 1.0, 0.753, 1.0), ambient=0.2),
    "director-purple": Material(color=(0.804, 0.6, 0.820, 1.0), ambient=0.2),
    "director-blue": Material(color=(0.207, 0.596, 0.792, 1.0), ambient=0.2),
    "none": None,
}
color_list = list(materials.keys())

def random_material():
    return Material(color=(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), 1), ambient=0.2)


class ViewerData(edict):
    """
    Interface to standardize viewer data.
    """

    def __init__(self, Rt, K, cols, rows, imgnames=None):
        self.imgnames = imgnames
        self.Rt = Rt
        self.K = K
        self.num_frames = Rt.shape[0]
        self.cols = cols
        self.rows = rows
        self.validate_format()

    def validate_format(self):
        assert len(self.Rt.shape) == 3
        assert self.Rt.shape[0] == self.num_frames
        assert self.Rt.shape[1] == 3
        assert self.Rt.shape[2] == 4

        assert len(self.K.shape) == 2
        assert self.K.shape[0] == 3
        assert self.K.shape[1] == 3
        if self.imgnames is not None:
            assert self.num_frames == len(self.imgnames)
            assert self.num_frames > 0
            im_p = self.imgnames[0]
            assert op.exists(im_p), f"Image path {im_p} does not exist"


class ARCTICViewer:
    def __init__(
        self,
        render_types=["rgb", "depth", "mask"],
        interactive=True,
        size=(2024, 2024),
        v = None,
    ):

        if v is None:
            if not interactive:
                v = HeadlessRenderer(size=size)
            else:
                v = Viewer(size=size)

        self.v = v
        self.interactive = interactive
        # self.layers = layers
        self.render_types = render_types

    def view_interactive(self):
        self.v.run()

    def view_fn_headless(self, num_iter, out_folder, video_name="video"):
        v = self.v

        # # 确保清理任何可能已经存在的资源
        # if hasattr(v, 'scene') and v.scene is not None:
        #     v.scene.release()  # 如果有这个方法的话

        v._init_scene()


        # print("self.render_types", self.render_types)
        # breakpoint()
        logger.info("Rendering to video")
        if "video" in self.render_types:
            vid_p = op.join(out_folder, f"{video_name}.mp4")
            # print("vid_p", vid_p)
            v.save_video(video_dir=vid_p, ensure_no_overwrite=False)
            return 
        # print("num_iter", num_iter)
        pbar = tqdm(range(num_iter))
        for fidx in pbar:
            out_rgb = op.join(out_folder, "images", f"rgb/{fidx:04d}.png")
            out_mask = op.join(out_folder, "images", f"mask/{fidx:04d}.png")
            out_depth = op.join(out_folder, "images", f"depth/{fidx:04d}.npy")

            # render RGB, depth, segmentation masks
            if "rgb" in self.render_types:
                v.export_frame(out_rgb)
            if "depth" in self.render_types:
                os.makedirs(op.dirname(out_depth), exist_ok=True)
                render_depth(v, out_depth)
            if "mask" in self.render_types:
                os.makedirs(op.dirname(out_mask), exist_ok=True)
                render_mask(v, out_mask)
            v.scene.next_frame()
        logger.info(f"Exported to {out_folder}")

    @abstractmethod
    def load_data(self):
        pass

    def check_format(self, batch):
        meshes_all, data = batch
        assert isinstance(meshes_all, dict)
        assert len(meshes_all) > 0
        for mesh in meshes_all.values():
            assert isinstance(mesh, Meshes)
        assert isinstance(data, ViewerData)

    def render_seq(self, batch, out_folder="./render_out", floor_y=0, video_name="video"):
        meshes_all, data = batch
        self.setup_viewer(data, floor_y)

        # breakpoint()
        for mesh in meshes_all.values():
            self.v.scene.add(mesh)
        # breakpoint()
        if self.interactive:
            print("self.interactive", self.interactive)
            self.view_interactive()
        else:
            print("self.interactive", self.interactive)
            num_iter = data["num_frames"]
            self.view_fn_headless(num_iter, out_folder, video_name)
        # breakpoint()

    def setup_viewer(self, data, floor_y):
        v = self.v
        fps = 30
        if "imgnames" in data:
            setup_billboard(data, v)

        # v.scene.background_color = (0.0, 0.0, 0.0, 1.0)

        # camera.show_path()
        v.run_animations = True  # autoplay
        v.run_animations = False  # autoplay
        v.playback_fps = fps
        v.scene.fps = fps
        v.scene.origin.enabled = False
        v.scene.floor.enabled = False
        v.auto_set_floor = False
        # v.scene.camera.position = np.array((0.0, 0.0, 0))
        self.v = v


def dist2vc(dist_ro, dist_lo, dist_o, _cmap, tf_fn=None):
    if tf_fn is not None:
        exp_map = tf_fn
    else:
        exp_map = small_exp_map
    dist_ro = exp_map(dist_ro)
    dist_lo = exp_map(dist_lo)
    dist_o = exp_map(dist_o)

    vc_ro = _cmap(dist_ro)
    vc_lo = _cmap(dist_lo)
    vc_o = _cmap(dist_o)
    return vc_ro, vc_lo, vc_o


def small_exp_map(_dist):
    dist = np.copy(_dist)
    # dist = 1.0 - np.clip(dist, 0, 0.1) / 0.1
    dist = np.exp(-20.0 * dist)
    return dist

def construct_viewer_meshes_random_color(data, draw_edges=False, flat_shading=True):
    rotation_flip = aa2rot_numpy(np.array([1, 0, 0]) * np.pi)
    meshes = {}
    for key, val in data.items():
        if 'single' in key:
            draw_edges = True
        else:
            draw_edges = False
        if "object" in key:
            flat_shading = False
        else:
            flat_shading = False

        v3d = val["v3d"]
        mesh_material = None  # 默认没有特定材质
        vertex_colors_data = val.get("vc", None) # 获取顶点颜色数据
        face_colors_data = val.get("fc", None) # 获取面片颜色数据

        # 只有在没有提供顶点颜色时，才尝试根据 "color" 键设置材质
        if vertex_colors_data is None:
            color_setting = val.get("color")
            if isinstance(color_setting, str):
                if color_setting == "random":
                    mesh_material = random_material()
                elif color_setting in materials:
                    mesh_material = materials[color_setting]
                # else: # 可以添加一个默认颜色或错误处理
                #    mesh_material = materials["white"]
            elif isinstance(color_setting, int) and color_setting == -1:
                 # 特殊情况，例如地面使用面片颜色，不需要设置材质
                 pass
            # else: # 处理其他未预期的 color_setting 类型
            #    mesh_material = materials["white"]

        # 创建 Meshes 对象
        meshes[key] = Meshes(
            v3d,
            val["f3d"],
            vertex_colors=vertex_colors_data,  # 传递顶点颜色
            face_colors=face_colors_data,      # 传递面片颜色
            name=val["name"],
            flat_shading=flat_shading,
            draw_edges=draw_edges,
            material=mesh_material,         # 传递材质 (如果顶点颜色存在，材质的基础色会被覆盖)
            # rotation=rotation_flip,
        )
    return meshes

def construct_viewer_meshes(data, draw_edges=False, flat_shading=True):
    rotation_flip = aa2rot_numpy(np.array([1, 0, 0]) * np.pi)
    meshes = {}
    for key, val in data.items():
        if 'single' in key:
            draw_edges = True
        else:
            draw_edges = False
        if "object" in key:
            flat_shading = False
        else:
            flat_shading = False
        v3d = val["v3d"]
        if not isinstance(val["color"], str):
            val["color"] = color_list[val["color"]]
        if val["color"] == "random":
            mesh_material = random_material()
        else:
            mesh_material = materials[val["color"]]
        meshes[key] = Meshes(
            v3d,
            val["f3d"],
            vertex_colors=val["vc"],
            face_colors=val["fc"] if "fc" in val else None,
            name=val["name"],
            flat_shading=flat_shading,
            draw_edges=draw_edges,
            material=mesh_material,
            # rotation=rotation_flip,
        )
    return meshes


def setup_viewer(
    v, shared_folder_p, video, images_path, data, flag, seq_name, side_angle
):
    fps = 10
    cols, rows = 224, 224
    focal = 1000.0

    # setup image paths
    regex = re.compile(r"(\d*)$")

    def sort_key(x):
        name = os.path.splitext(x)[0]
        return int(regex.search(name).group(0))

    # setup billboard
    images_path = op.join(shared_folder_p, "images")
    images_paths = [
        os.path.join(images_path, f)
        for f in sorted(os.listdir(images_path), key=sort_key)
    ]
    assert len(images_paths) > 0

    cam_t = data[f"{flag}.object.cam_t"]
    num_frames = min(cam_t.shape[0], len(images_paths))
    cam_t = cam_t[:num_frames]
    # setup camera
    K = np.array([[focal, 0, rows / 2.0], [0, focal, cols / 2.0], [0, 0, 1]])
    Rt = np.zeros((num_frames, 3, 4))
    Rt[:, :, 3] = cam_t
    Rt[:, :3, :3] = np.eye(3)
    Rt[:, 1:3, :3] *= -1.0

    camera = OpenCVCamera(K, Rt, cols, rows, viewer=v)
    if side_angle is None:
        billboard = Billboard.from_camera_and_distance(
            camera, 10.0, cols, rows, images_paths
        )
        v.scene.add(billboard)
    v.scene.add(camera)
    v.run_animations = True  # autoplay
    v.playback_fps = fps
    v.scene.fps = fps
    v.scene.origin.enabled = False
    v.scene.floor.enabled = False
    v.auto_set_floor = False
    v.scene.floor.position[1] = -3
    v.set_temp_camera(camera)
    # v.scene.camera.position = np.array((0.0, 0.0, 0))
    return v


def render_depth(v, depth_p):
    depth = np.array(v.get_depth()).astype(np.float16)
    np.save(depth_p, depth)


def render_mask(v, mask_p):
    nodes_uid = {node.name: node.uid for node in v.scene.collect_nodes()}
    my_cmap = {
        uid: [SEGM_IDS[name], SEGM_IDS[name], SEGM_IDS[name]]
        for name, uid in nodes_uid.items()
        if name in SEGM_IDS.keys()
    }
    mask = np.array(v.get_mask(color_map=my_cmap)).astype(np.uint8)
    mask = Image.fromarray(mask)
    mask.save(mask_p)


def setup_billboard(data, v):
    images_paths = data.imgnames
    K = data.K
    Rt = data.Rt
    rows = data.rows
    cols = data.cols
    camera = OpenCVCamera(K, Rt, cols, rows, viewer=v)
    if images_paths is not None:
        billboard = Billboard.from_camera_and_distance(
            camera, 10.0, cols, rows, images_paths
        )
        v.scene.add(billboard)
    v.scene.add(camera)

    # # 修改这里 - 检查相机对象是否有 load_cam 方法
    # if hasattr(v.scene.camera, 'load_cam'):
    #     v.scene.camera.load_cam()
    v.scene.camera.load_cam()
    v.set_temp_camera(camera)
