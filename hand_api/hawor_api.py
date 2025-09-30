from lib.pipeline.tools import detect_track
import os
from natsort import natsorted
from glob import glob
import numpy as np

def detect_track_video(img_folder):

    seq_folder = img_folder.replace("images/images", "images/reconstruction")
    os.makedirs(seq_folder, exist_ok=True)
    os.makedirs(img_folder, exist_ok=True)
    print(f'Running detect_track on {img_folder} ...')

    ##### Extract Frames #####
    imgfiles = natsorted(glob(f'{img_folder}/*.png'))
    # print(imgfiles)

    ##### Detection + Track #####
    print('Detect and Track ...')

    start_idx = 0
    end_idx = len(imgfiles)

    if os.path.exists(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_boxes.npy'):
        print(f"skip track for {start_idx}_{end_idx}")
        return start_idx, end_idx, seq_folder, imgfiles
    os.makedirs(f"{seq_folder}/tracks_{start_idx}_{end_idx}", exist_ok=True)
    #breakpoint()
    boxes_, tracks_ = detect_track(imgfiles, thresh=0.2)
    np.save(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_boxes.npy', boxes_)
    np.save(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_tracks.npy', tracks_)

    return start_idx, end_idx, seq_folder, imgfiles


from collections import defaultdict
from lib.models.hawor import HAWOR
import torch
import cv2
from lib.vis.renderer import Renderer
from lib.eval_utils.custom_utils import interpolate_bboxes
from lib.pipeline.tools import parse_chunks, parse_chunks_hand_frame
import json
import joblib
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from hawor.utils.rotation import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis

CHECKPOINT = "./weights/hawor/checkpoints/hawor.ckpt"

def load_hawor(checkpoint_path):
    from pathlib import Path
    from hawor.configs import get_config
    # print("load hawor model 1")
    model_cfg = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')
    model_cfg = get_config(model_cfg, update_cachedir=True)

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192,256]
        model_cfg.freeze()
    # print("load hawor model 2")
    model = HAWOR.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    # breakpoint()
    return model, model_cfg

def hawor_motion_estimation(img_folder, start_idx, end_idx, seq_folder, img_focal=None):
    print("in hawor_motion_estimation")
    # breakpoint()
    model, model_cfg = load_hawor(CHECKPOINT)
    print("load hawor model")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # img_folder = f"{video_root}/{video}/extracted_images"
    # imgfiles = np.array(natsorted(glob(f'{img_folder}/*.jpg')))
    imgfiles = np.array(natsorted(glob(f'{img_folder}/*.png')))

    tracks = np.load(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_tracks.npy', allow_pickle=True).item()
    if img_focal is None:
        try:
            with open(os.path.join(seq_folder, 'est_focal.txt'), 'r') as file:
                img_focal = file.read()
                img_focal = float(img_focal)
        except:
            #需要修改成当前相机的
            img_focal = 387.8267517089844
            print(f'No focal length provided, use default {img_focal}')
            with open(os.path.join(seq_folder, 'est_focal.txt'), 'w') as file:
                file.write(str(img_focal))
    
    tid = np.array([tr for tr in tracks])

    if os.path.exists(f'{seq_folder}/tracks_{start_idx}_{end_idx}/frame_chunks_all.npy'):
        print("skip hawor motion estimation")
        frame_chunks_all = joblib.load(f'{seq_folder}/tracks_{start_idx}_{end_idx}/frame_chunks_all.npy')
        return frame_chunks_all, img_focal

    print(f'Running hawor on {img_folder} ...')

    left_trk = []
    right_trk = []
    for k, idx in enumerate(tid):
        trk = tracks[idx]

        valid = np.array([t['det'] for t in trk])        
        is_right = np.concatenate([t['det_handedness'] for t in trk])[valid]
        
        if is_right.sum() / len(is_right) < 0.5:
            left_trk.extend(trk)
        else:
            right_trk.extend(trk)
    left_trk = sorted(left_trk, key=lambda x: x['frame'])
    right_trk = sorted(right_trk, key=lambda x: x['frame'])
    final_tracks = {
        0: left_trk,
        1: right_trk
    }
    tid = [0, 1]

    # print(imgfiles)
    img = cv2.imread(imgfiles[0])
    img_center = [img.shape[1] / 2, img.shape[0] / 2]# w/2, h/2  
    H, W = img.shape[:2]
    model_masks = np.zeros((len(imgfiles), H, W))

    bin_size = 128
    max_faces_per_bin = 20000
    renderer = Renderer(img.shape[1], img.shape[0], img_focal, 'cuda', 
                    bin_size=bin_size, max_faces_per_bin=max_faces_per_bin)
    # get faces
    faces = get_mano_faces()
    faces_new = np.array([[92, 38, 234],
            [234, 38, 239],
            [38, 122, 239],
            [239, 122, 279],
            [122, 118, 279],
            [279, 118, 215],
            [118, 117, 215],
            [215, 117, 214],
            [117, 119, 214],
            [214, 119, 121],
            [119, 120, 121],
            [121, 120, 78],
            [120, 108, 78],
            [78, 108, 79]])
    faces_right = np.concatenate([faces, faces_new], axis=0)
    faces_left = faces_right[:,[0,2,1]]

    frame_chunks_all = defaultdict(list)
    for idx in tid:
        print(f"tracklet {idx}:")
        trk = final_tracks[idx]

        # interp bboxes
        valid = np.array([t['det'] for t in trk])
        if valid.sum() < 2:
            continue
        boxes = np.concatenate([t['det_box'] for t in trk])
        non_zero_indices = np.where(np.any(boxes != 0, axis=1))[0]
        first_non_zero = non_zero_indices[0]
        last_non_zero = non_zero_indices[-1]
        boxes[first_non_zero:last_non_zero+1] = interpolate_bboxes(boxes[first_non_zero:last_non_zero+1])
        valid[first_non_zero:last_non_zero+1] = True


        boxes = boxes[first_non_zero:last_non_zero+1]
        is_right = np.concatenate([t['det_handedness'] for t in trk])[valid]
        frame = np.array([t['frame'] for t in trk])[valid]
        
        if is_right.sum() / len(is_right) < 0.5:
            is_right = np.zeros((len(boxes), 1))
        else:
            is_right = np.ones((len(boxes), 1))

        frame_chunks, boxes_chunks = parse_chunks(frame, boxes, min_len=1)
        frame_chunks_all[idx] = frame_chunks

        if len(frame_chunks) == 0:
            continue

        for frame_ck, boxes_ck in zip(frame_chunks, boxes_chunks):
            print(f"inference from frame {frame_ck[0]} to {frame_ck[-1]}")
            img_ck = imgfiles[frame_ck]
            if is_right[0] > 0:
                do_flip = False
            else:
                do_flip = True
                
            results = model.inference(img_ck, boxes_ck, img_focal=img_focal, img_center=img_center, do_flip=do_flip)

            data_out = {
                "init_root_orient": results["pred_rotmat"][None, :, 0], # (B, T, 3, 3)
                "init_hand_pose": results["pred_rotmat"][None, :, 1:], # (B, T, 15, 3, 3)
                "init_trans": results["pred_trans"][None, :, 0],  # (B, T, 3)
                "init_betas": results["pred_shape"][None, :]  # (B, T, 10)
            }

            # flip left hand
            init_root = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
            init_hand_pose = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
            if do_flip:
                init_root[..., 1] *= -1
                init_root[..., 2] *= -1
                init_hand_pose[..., 1] *= -1
                init_hand_pose[..., 2] *= -1
            data_out["init_root_orient"] = angle_axis_to_rotation_matrix(init_root)
            data_out["init_hand_pose"] = angle_axis_to_rotation_matrix(init_hand_pose)

            # save camera-space results
            pred_dict={
                k:v.tolist() for k, v in data_out.items()
            }
            pred_path = os.path.join(seq_folder, 'cam_space', str(idx), f"{frame_ck[0]}_{frame_ck[-1]}.json")
            if not os.path.exists(os.path.join(seq_folder, 'cam_space', str(idx))):
                os.makedirs(os.path.join(seq_folder, 'cam_space', str(idx)))
            with open(pred_path, "w") as f:
                json.dump(pred_dict, f, indent=1)


            # get hand mask
            data_out["init_root_orient"] = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
            data_out["init_hand_pose"] = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
            if do_flip: # left
                outputs = run_mano_left(data_out["init_trans"], data_out["init_root_orient"], data_out["init_hand_pose"], betas=data_out["init_betas"])
            else: # right
                outputs = run_mano(data_out["init_trans"], data_out["init_root_orient"], data_out["init_hand_pose"], betas=data_out["init_betas"])
            
            vertices = outputs["vertices"][0].cpu()  # (T, N, 3)
            for img_i, _ in enumerate(img_ck):
                if do_flip:
                    faces = torch.from_numpy(faces_left).cuda()
                else:
                    faces = torch.from_numpy(faces_right).cuda()
                cam_R = torch.eye(3).unsqueeze(0).cuda()
                cam_T = torch.zeros(1, 3).cuda()
                cameras, lights = renderer.create_camera_from_cv(cam_R, cam_T)
                verts_color = torch.tensor([0, 0, 255, 255]) / 255
                vertices_i = vertices[[img_i]]
                rend, mask = renderer.render_multiple(vertices_i.unsqueeze(0).cuda(), faces, verts_color.unsqueeze(0).cuda(), cameras, lights)
                
                model_masks[frame_ck[img_i]] += mask
                
    model_masks = model_masks > 0 # bool
    np.save(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_masks.npy', model_masks)
    joblib.dump(frame_chunks_all, f'{seq_folder}/tracks_{start_idx}_{end_idx}/frame_chunks_all.npy')
    return frame_chunks_all, img_focal

from infiller.lib.model.network import TransformerModel
from lib.eval_utils.custom_utils import cam2world_convert, load_slam_cam
from lib.eval_utils.filling_utils import filling_postprocess, filling_preprocess

INFILLER_WEIGHT = "./weights/hawor/checkpoints/infiller.pt"
def hawor_infiller(img_folder, start_idx, end_idx, frame_chunks_all, seq_folder, extrinsics=None):
    # load infiller
    weight_path = INFILLER_WEIGHT
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ckpt = torch.load(weight_path, map_location=device)
    pos_dim = 3
    shape_dim = 10
    num_joints = 15
    rot_dim = (num_joints + 1) * 6 # rot6d
    repr_dim = 2 * (pos_dim + shape_dim + rot_dim)
    nhead = 8 # repr_dim = 154
    horizon = 120
    filling_model = TransformerModel(seq_len=horizon, input_dim=repr_dim, d_model=384, nhead=nhead, d_hid=2048, nlayers=8, dropout=0.05, out_dim=repr_dim, masked_attention_stage=True)
    filling_model.to(device)
    filling_model.load_state_dict(ckpt['transformer_encoder_state_dict'])
    filling_model.eval()

    # file = args.video_path
    # video_root = os.path.dirname(file)
    # video = os.path.basename(file).split('.')[0]
    # seq_folder = os.path.join(video_root, video)
    # img_folder = f"{video_root}/{video}/extracted_images"

    # Previous steps
    # imgfiles = np.array(natsorted(glob(f'{img_folder}/*.jpg')))
    imgfiles = np.array(natsorted(glob(f'{img_folder}/*.png')))

    idx2hand = ['left', 'right']
    filling_length = 120

    if extrinsics is not None:
        # 世界坐标系到相机坐标系(w2c)
        R_w2c_sla = torch.from_numpy(extrinsics[:3, :3]).to(torch.float32)  # 旋转矩阵(3x3)
        t_w2c_sla = torch.from_numpy(extrinsics[:3, 3]).to(torch.float32)   # 平移向量(3x1)

        # 相机坐标系到世界坐标系(c2w)
        extrinsics_inverse = np.linalg.inv(extrinsics)
        R_c2w_sla = torch.from_numpy(extrinsics_inverse[:3, :3]).to(torch.float32)  # 旋转矩阵(3x3)
        t_c2w_sla = torch.from_numpy(extrinsics_inverse[:3, 3]).to(torch.float32)   # 平移向量(3x1)

        # 添加批次维度
        batch_size = len(imgfiles)  # 您需要的批次大小
        R_c2w_sla_all = R_c2w_sla.unsqueeze(0).expand(batch_size, 3, 3)  # 形状变为 [b,3,3]
        t_c2w_sla_all = t_c2w_sla.unsqueeze(0).expand(batch_size, 3)      # 形状变为 [b,3]

        R_w2c_sla_all = R_w2c_sla.unsqueeze(0).expand(batch_size, 3, 3)  # 形状变为 [b,3,3]
        t_w2c_sla_all = t_w2c_sla.unsqueeze(0).expand(batch_size, 3)      # 形状变为 [b,3]
        # print(R_c2w)
        # print(t_c2w)
        # breakpoint()
    else:
        fpath = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
        R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(fpath)
        # print(R_c2w_sla_all)
        # print(t_c2w_sla_all)
        # breakpoint()

    pred_trans = torch.zeros(2, len(imgfiles), 3)
    pred_rot = torch.zeros(2, len(imgfiles), 3)
    pred_hand_pose = torch.zeros(2, len(imgfiles), 45)
    pred_betas = torch.zeros(2, len(imgfiles), 10)
    pred_valid = torch.zeros((2, pred_betas.size(1)))    

    # camera space to world space
    tid = [0, 1]            
    for k, idx in enumerate(tid):
        frame_chunks = frame_chunks_all[idx]

        if len(frame_chunks) == 0:
            continue

        for frame_ck in frame_chunks:
            print(f"from frame {frame_ck[0]} to {frame_ck[-1]}")                
            pred_path = os.path.join(seq_folder, 'cam_space', str(idx), f"{frame_ck[0]}_{frame_ck[-1]}.json")
            with open(pred_path, "r") as f:
                pred_dict = json.load(f)
            # print(f"JSON格式中init_root_orient的形状: {np.array(pred_dict['init_root_orient']).shape}")
            data_out = {
                k:torch.tensor(v) for k, v in pred_dict.items()
                }

            # print(frame_ck)
            R_c2w_sla = R_c2w_sla_all[frame_ck]
            t_c2w_sla = t_c2w_sla_all[frame_ck]

            data_world = cam2world_convert(R_c2w_sla, t_c2w_sla, data_out, 'right' if idx > 0 else 'left')

            pred_trans[[idx], frame_ck] = data_world["init_trans"]
            pred_rot[[idx], frame_ck] = data_world["init_root_orient"]
            pred_hand_pose[[idx], frame_ck] = data_world["init_hand_pose"].flatten(-2)
            pred_betas[[idx], frame_ck] = data_world["init_betas"]
            pred_valid[[idx], frame_ck] = 1
            
        
    # runing fillingnet for this video
    frame_list = torch.tensor(list(range(pred_trans.size(1))))
    pred_valid = (pred_valid > 0).numpy()
    for k, idx in enumerate([1, 0]):
        missing = ~pred_valid[idx]

        frame = frame_list[missing]
        frame_chunks = parse_chunks_hand_frame(frame)

        print(f"run infiller on {idx2hand[idx]} hand ...")
        for frame_ck in tqdm(frame_chunks):
            start_shift = -1
            while frame_ck[0] + start_shift >= 0 and pred_valid[:, frame_ck[0] + start_shift].sum() != 2:
                start_shift -= 1  # Shift to find the previous valid frame as start
            print(f"run infiller on frame {frame_ck[0] + start_shift} to frame {min(len(imgfiles)-1, frame_ck[0] + start_shift + filling_length)}")

            frame_start = frame_ck[0]
            filling_net_start = max(0, frame_start + start_shift)
            filling_net_end = min(len(imgfiles)-1, filling_net_start + filling_length)
            seq_valid = pred_valid[:, filling_net_start:filling_net_end]
            filling_seq = {}
            filling_seq['trans'] = pred_trans[:, filling_net_start:filling_net_end].numpy()
            filling_seq['rot'] = pred_rot[:, filling_net_start:filling_net_end].numpy()
            filling_seq['hand_pose'] = pred_hand_pose[:, filling_net_start:filling_net_end].numpy()
            filling_seq['betas'] = pred_betas[:, filling_net_start:filling_net_end].numpy()
            filling_seq['valid'] = seq_valid
            # preprocess (convert to canonical + slerp)
            filling_input, transform_w_canon = filling_preprocess(filling_seq)
            src_mask = torch.zeros((filling_length, filling_length), device=device).type(torch.bool)
            src_mask = src_mask.to(device)
            filling_input = torch.from_numpy(filling_input).unsqueeze(0).to(device).permute(1,0,2) # (seq_len, B, in_dim)
            T_original = len(filling_input)
            filling_length = 120
            if T_original < filling_length:
                pad_length = filling_length - T_original
                last_time_step = filling_input[-1, :, :]
                padding = last_time_step.unsqueeze(0).repeat(pad_length, 1, 1)
                filling_input = torch.cat([filling_input, padding], dim=0) 
                seq_valid_padding = np.ones((2, filling_length - T_original))
                seq_valid_padding = np.concatenate([seq_valid, seq_valid_padding], axis=1) 
            else:
                seq_valid_padding = seq_valid
                

            T, B, _ = filling_input.shape

            valid = torch.from_numpy(seq_valid_padding).unsqueeze(0).all(dim=1).permute(1, 0) # (T,B)
            valid_atten = torch.from_numpy(seq_valid_padding).unsqueeze(0).all(dim=1).unsqueeze(1) # (B,1,T)
            data_mask = torch.zeros((horizon, B, 1), device=device, dtype=filling_input.dtype)
            data_mask[valid] = 1
            atten_mask = torch.ones((B, 1, horizon),
                        device=device, dtype=torch.bool)
            atten_mask[valid_atten] = False
            atten_mask = atten_mask.unsqueeze(2).repeat(1, 1, T, 1) # (B,1,T,T)

            output_ck = filling_model(filling_input, src_mask, data_mask, atten_mask)

            output_ck = output_ck.permute(1,0,2).reshape(T, 2, -1).cpu().detach() #  two hands

            output_ck = output_ck[:T_original]

            filling_output = filling_postprocess(output_ck, transform_w_canon)

            # repalce the missing prediciton with infiller output
            filling_seq['trans'][~seq_valid] = filling_output['trans'][~seq_valid]
            filling_seq['rot'][~seq_valid] = filling_output['rot'][~seq_valid]
            filling_seq['hand_pose'][~seq_valid] = filling_output['hand_pose'][~seq_valid]
            filling_seq['betas'][~seq_valid] = filling_output['betas'][~seq_valid]

            pred_trans[:, filling_net_start:filling_net_end] = torch.from_numpy(filling_seq['trans'][:])
            pred_rot[:, filling_net_start:filling_net_end] = torch.from_numpy(filling_seq['rot'][:])
            pred_hand_pose[:, filling_net_start:filling_net_end] = torch.from_numpy(filling_seq['hand_pose'][:])
            pred_betas[:, filling_net_start:filling_net_end] = torch.from_numpy(filling_seq['betas'][:])
            pred_valid[:, filling_net_start:filling_net_end] = 1
    save_path = os.path.join(seq_folder, "world_space_res.pth")
    joblib.dump([pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid], save_path)
    return pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid


from lib.pipeline.masked_droid_slam import *
from lib.pipeline.est_scale import *
from hawor.utils.process import block_print, enable_print
sys.path.insert(0, os.path.dirname(__file__) + '/thirdparty/Metric3D')
# print(os.path.dirname(__file__))
# sys.path.insert(0, os.path.dirname(__file__) + 'thirdparty/Metric3D')
from metric import Metric3D
import math

def hawor_slam(img_folder, start_idx, end_idx, seq_folder, img_focal=None, enable_camera_jitter=True):
    # File and folders
    # file = args.video_path
    # video_root = os.path.dirname(file)
    # video = os.path.basename(file).split('.')[0]
    # seq_folder = os.path.join(video_root, video)
    # os.makedirs(seq_folder, exist_ok=True)
    # seq_folder = os.path.join(video_root, video)

    # img_folder = f'{seq_folder}/extracted_images'
    # imgfiles = natsorted(glob(f'{img_folder}/*.jpg'))
    imgfiles = natsorted(glob(f'{img_folder}/*.png'))

    first_img = cv2.imread(imgfiles[0])
    height, width, _ = first_img.shape
    
    print(f'Running slam on {seq_folder} ...')
    if enable_camera_jitter:
        print('启用相机抖动模式以增强SLAM稳定性')

    ##### Run SLAM #####
    # Use Masking
    masks = np.load(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_masks.npy', allow_pickle=True)
    masks = torch.from_numpy(masks)
    print(masks.shape)

    # Camera calibration (intrinsics) for SLAM
    # focal = args.img_focal
    focal = img_focal
    if focal is None:
        try:
            with open(os.path.join(seq_folder, 'est_focal.txt'), 'r') as file:
                focal = file.read()
                focal = float(focal)
        except:
            
            print('No focal length provided')
            focal = 600
            with open(os.path.join(seq_folder, 'est_focal.txt'), 'w') as file:
                file.write(str(focal))
    calib = np.array(est_calib(imgfiles)) # [focal, focal, cx, cy]
    center = calib[2:]        
    calib[:2] = focal
    
    # Droid-slam with masking and optional camera jitter
    slam_success = False
    final_error = None
    
    # 先尝试不使用相机抖动
    try:
        print("尝试不使用相机抖动运行SLAM...")
        droid, traj = run_slam(imgfiles, masks=masks, calib=calib, enable_camera_jitter=False)
        slam_success = True
        print("✓ SLAM without camera jitter succeeded")
    except Exception as e:
        print(f"✗ SLAM without camera jitter failed: {e}")
        final_error = e

    # 如果失败则尝试使用相机抖动
    if not slam_success and enable_camera_jitter:
        try:
            print("尝试使用相机抖动运行SLAM...")
            droid, traj = run_slam(imgfiles, masks=masks, calib=calib, enable_camera_jitter=True)
            slam_success = True
            print("✓ SLAM with camera jitter succeeded")
        except Exception as e:
            print(f"✗ SLAM with camera jitter failed: {e}")
            final_error = e
    
    
    if not slam_success:
        print(f"所有SLAM尝试都失败了，最后错误: {final_error}")
        raise final_error
    n = droid.video.counter.value
    tstamp = droid.video.tstamp.cpu().int().numpy()[:n]
    disps = droid.video.disps_up.cpu().numpy()[:n]
    print('DBA errors:', droid.backend.errors)

    del droid
    torch.cuda.empty_cache()

    # Estimate scale  
    block_print()  
    metric = Metric3D('thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth') 
    enable_print() 
    min_threshold = 0.4
    max_threshold = 0.7

    print('Predicting Metric Depth ...')
    pred_depths = []
    H, W = get_dimention(imgfiles)
    for t in tqdm(tstamp):
        pred_depth = metric(imgfiles[t], calib)
        pred_depth = cv2.resize(pred_depth, (W, H))
        pred_depths.append(pred_depth)

    ##### Estimate Metric Scale #####
    print('Estimating Metric Scale ...')
    scales_ = []
    n = len(tstamp)   # for each keyframe
    for i in tqdm(range(n)):
        t = tstamp[i]
        disp = disps[i]
        pred_depth = pred_depths[i]
        slam_depth = 1/disp
        
        # Estimate scene scale
        msk = masks[t].numpy().astype(np.uint8)
        scale = est_scale_hybrid(slam_depth, pred_depth, sigma=0.5, msk=msk, near_thresh=min_threshold, far_thresh=max_threshold)  
        while math.isnan(scale):
            min_threshold -= 0.1
            max_threshold += 0.1
            scale = est_scale_hybrid(slam_depth, pred_depth, sigma=0.5, msk=msk, near_thresh=min_threshold, far_thresh=max_threshold)                    
        scales_.append(scale)

    median_s = np.median(scales_)
    print(f"estimated scale: {median_s}")

    # Save results
    os.makedirs(f"{seq_folder}/SLAM", exist_ok=True)
    save_path = f'{seq_folder}/SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz'
    np.savez(save_path, 
            tstamp=tstamp, disps=disps, traj=traj, 
            img_focal=focal, img_center=calib[-2:],
            scale=median_s)    


def compute_all_finger_distances(joints):
    """
    计算MANO模型中大拇指末端到其他所有手指末端的距离
    
    参数:
    joints: torch.Tensor - MANO模型输出的关节点坐标
            形状可以是 [batch_size, time_steps, 21, 3] 或 [time_steps, 21, 3]
    
    返回:
    包含所有距离的字典，每个距离的形状为 [time_steps, 1]
    """
    # 处理输入维度
    if len(joints.shape) == 3:  # [time_steps, 21, 3]
        joints = joints.unsqueeze(0)  # 添加批次维度 [1, time_steps, 21, 3]
    
    batch_size, time_steps, num_joints, _ = joints.shape
    
    # MANO模型中的关节索引
    wrist_idx = 0       # 手腕根节点索引
    thumb_mcp_idx = 1   # 大拇指掌指关节索引
    thumb_pip_idx = 2   # 大拇指近指间关节索引
    thumb_dip_idx = 3   # 大拇指远指间关节索引
    thumb_tip_idx = 4   # 大拇指末端节点索引
    index_mcp_idx = 5   # 食指掌指关节索引
    index_pip_idx = 6   # 食指近指间关节索引
    index_dip_idx = 7   # 食指远指间关节索引
    index_tip_idx = 8   # 食指末端节点索引
    middle_mcp_idx = 9  # 中指掌指关节索引
    middle_pip_idx = 10 # 中指近指间关节索引
    middle_dip_idx = 11 # 中指远指间关节索引
    middle_tip_idx = 12 # 中指末端节点索引
    ring_mcp_idx = 13   # 无名指掌指关节索引
    ring_pip_idx = 14   # 无名指近指间关节索引
    ring_dip_idx = 15   # 无名指远指间关节索引
    ring_tip_idx = 16   # 无名指末端节点索引
    pinky_mcp_idx = 17  # 小拇指掌指关节索引
    pinky_pip_idx = 18  # 小拇指近指间关节索引
    pinky_dip_idx = 19  # 小拇指远指间关节索引
    pinky_tip_idx = 20  # 小拇指末端节点索引
    
    # 获取各指尖坐标
    thumb_tip = joints[:, :, thumb_tip_idx]   # [batch_size, time_steps, 3]
    index_tip = joints[:, :, index_tip_idx]   # [batch_size, time_steps, 3]
    middle_tip = joints[:, :, middle_tip_idx]  # [batch_size, time_steps, 3]
    ring_tip = joints[:, :, ring_tip_idx]     # [batch_size, time_steps, 3]
    pinky_tip = joints[:, :, pinky_tip_idx]   # [batch_size, time_steps, 3]
    
    # 计算欧几里得距离
    thumb_to_index_distance = torch.norm(thumb_tip - index_tip, dim=2).view(time_steps, 1)
    thumb_to_middle_distance = torch.norm(thumb_tip - middle_tip, dim=2).view(time_steps, 1)
    thumb_to_ring_distance = torch.norm(thumb_tip - ring_tip, dim=2).view(time_steps, 1)
    thumb_to_pinky_distance = torch.norm(thumb_tip - pinky_tip, dim=2).view(time_steps, 1)
    
    return {
        "thumb_to_index_distance": thumb_to_index_distance,
        "thumb_to_middle_distance": thumb_to_middle_distance,
        "thumb_to_ring_distance": thumb_to_ring_distance,
        "thumb_to_pinky_distance": thumb_to_pinky_distance
    }

def computer_thumb_index_thenar_midpoint(joints):
    """
    计算大拇指近指间关节和食指掌指关节中点
    
    MANO Joint Index Reference:
    0: wrist, 1-4: thumb (1=CMC, 2=MCP, 3=IP, 4=tip), 5-8: index (5=MCP, 6=PIP, 7=DIP, 8=tip),
    9-12: middle (9=MCP, 10=PIP, 11=DIP, 12=tip), 13-16: ring (13=MCP, 14=PIP, 15=DIP, 16=tip),
    17-20: pinky (17=MCP, 18=PIP, 19=DIP, 20=tip)
    """
    if len(joints.shape) == 3:  # [time_steps, 21, 3]
        joints = joints.unsqueeze(0)  # 添加批次维度 [1, time_steps, 21, 3]
    
    batch_size, time_steps, num_joints, _ = joints.shape

    thumb_pip_idx = 2   # 大拇指近指间关节索引 (thumb MCP)
    index_mcp_idx = 5   # 食指掌指关节索引 (index MCP)
    thumb_joint = joints[:, :, thumb_pip_idx]   # [batch_size, time_steps, 3]
    index_joint = joints[:, :, index_mcp_idx]   # [batch_size, time_steps, 3] 
    
    print(f"DEBUG thenar midpoint calculation:")
    print(f"  Thumb MCP (joint {thumb_pip_idx}): {thumb_joint[0,0] if thumb_joint.numel() > 0 else 'empty'}")
    print(f"  Index MCP (joint {index_mcp_idx}): {index_joint[0,0] if index_joint.numel() > 0 else 'empty'}")
    
    # 计算食指和拇指中点
    midpoint = (thumb_joint + index_joint) / 2
    midpoint = midpoint.squeeze(0)
    print(f"  Computed midpoint: {midpoint[0] if midpoint.numel() > 0 else 'empty'}")
    print(f"midpoint.shape:{midpoint.shape}")
    return midpoint

def computer_thumb_index_tip_midpoint(joints):
    """
    计算食指和拇指指尖中点
    """
    if len(joints.shape) == 3:  # [time_steps, 21, 3]
        joints = joints.unsqueeze(0)  # 添加批次维度 [1, time_steps, 21, 3]
    
    batch_size, time_steps, num_joints, _ = joints.shape

    thumb_tip_idx = 4   # 大拇指末端节点索引
    index_tip_idx = 8   # 食指末端节点索引

    thumb_tip = joints[:, :, thumb_tip_idx]   # [batch_size, time_steps, 3]
    index_tip = joints[:, :, index_tip_idx]   # [batch_size, time_steps, 3]
    # 计算食指和拇指指尖中点
    midpoint = (thumb_tip + index_tip) / 2
    midpoint = midpoint.squeeze(0)
    return midpoint
