import torch
from lib.models.mano_wrapper import MANO
from hawor.utils.geometry import aa_to_rotmat
import numpy as np
import sys
import os

def block_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__

def get_mano_faces():
    block_print()
    MANO_cfg = {
        'DATA_DIR': '_DATA/data/',
        'MODEL_PATH': '_DATA/data/mano',
        'GENDER': 'neutral',
        'NUM_HAND_JOINTS': 15,
        'CREATE_BODY_POSE': False
    }
    mano_cfg = {k.lower(): v for k,v in MANO_cfg.items()}
    mano = MANO(**mano_cfg)
    enable_print()
    return mano.faces


def run_mano(trans, root_orient, hand_pose, is_right=None, betas=None, use_cuda=True):
    """
    Forward pass of the SMPL model and populates pred_data accordingly with
    joints3d, verts3d, points3d.

    trans : B x T x 3
    root_orient : B x T x 3
    body_pose : B x T x J*3
    betas : (optional) B x D
    """
    print("=" * 50)
    print("MANO FUNCTION INPUT DEBUG")  
    print("=" * 50)
    print(f"Input trans shape: {trans.shape}, sample: {trans[0,0] if trans.numel() > 0 else 'empty'}")
    print(f"Input root_orient shape: {root_orient.shape}, sample: {root_orient[0,0] if root_orient.numel() > 0 else 'empty'}")
    print(f"Input hand_pose shape: {hand_pose.shape}")
    if betas is not None:
        print(f"Input betas shape: {betas.shape}, sample: {betas[0,0] if betas.numel() > 0 else 'empty'}")
    print("=" * 50)
    
    # block_print()
    MANO_cfg = {
        'DATA_DIR': '_DATA/data/',
        'MODEL_PATH': '_DATA/data/mano',
        'GENDER': 'neutral',
        'NUM_HAND_JOINTS': 15,
        'CREATE_BODY_POSE': False
    }
    mano_cfg = {k.lower(): v for k,v in MANO_cfg.items()}
    mano = MANO(**mano_cfg)
    if use_cuda:
        mano = mano.cuda()
    
    # skinning_weights = mano.lbs_weights
    # if skinning_weights is not None:
    #     understand_skinning_weights(skinning_weights)
    # else:
    #     print("Warning: 无法访问skinning weights，跳过分析")

    B, T, _ = root_orient.shape
    NUM_JOINTS = 15
    mano_params = {
        'global_orient': root_orient.reshape(B*T, -1),
        'hand_pose': hand_pose.reshape(B*T*NUM_JOINTS, 3),
        'betas': betas.reshape(B*T, -1),
    }
    rotmat_mano_params = mano_params
    rotmat_mano_params['global_orient'] = aa_to_rotmat(mano_params['global_orient']).view(B*T, 1, 3, 3)
    rotmat_mano_params['hand_pose'] = aa_to_rotmat(mano_params['hand_pose']).view(B*T, NUM_JOINTS, 3, 3)
    rotmat_mano_params['transl'] = trans.reshape(B*T, 3)

    if use_cuda:
        mano_output = mano(**{k: v.float().cuda() for k,v in rotmat_mano_params.items()}, pose2rot=False)
    else:
        mano_output = mano(**{k: v.float() for k,v in rotmat_mano_params.items()}, pose2rot=False)

    faces_right = mano.faces
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
    faces_right = np.concatenate([faces_right, faces_new], axis=0)
    faces_n = len(faces_right)
    faces_left = faces_right[:,[0,2,1]]
    
    outputs = {
        "joints": mano_output.joints.reshape(B, T, -1, 3),
        "vertices": mano_output.vertices.reshape(B, T, -1, 3),
    }

    if not is_right is None:
        # outputs["vertices"][..., 0] = (2*is_right-1)*outputs["vertices"][..., 0]
        # outputs["joints"][..., 0] = (2*is_right-1)*outputs["joints"][..., 0]
        is_right = (is_right[:, :, 0].cpu().numpy() > 0)
        faces_result = np.zeros((B, T, faces_n, 3))
        faces_right_expanded = np.expand_dims(np.expand_dims(faces_right, axis=0), axis=0) 
        faces_left_expanded = np.expand_dims(np.expand_dims(faces_left, axis=0), axis=0) 
        faces_result = np.where(is_right[..., np.newaxis, np.newaxis], faces_right_expanded, faces_left_expanded)
        outputs["faces"] = torch.from_numpy(faces_result.astype(np.int32))

    # DEBUG: Print MANO raw output information
    print("=" * 50)
    print("MANO RAW OUTPUT DEBUG")  
    print("=" * 50)
    print(f"MANO output joints shape: {outputs['joints'].shape}")
    print(f"MANO output vertices shape: {outputs['vertices'].shape}")
    print(f"MANO joints[0,0] (first batch, first frame, all joints xyz):\n{outputs['joints'][0,0]}")
    print(f"MANO translation for first frame: {trans[0,0]}")
    print(f"MANO root orientation for first frame: {root_orient[0,0]}")
    print(f"First few joint positions from MANO:")
    for i in range(min(5, outputs['joints'].shape[2])):
        print(f"  Joint {i}: {outputs['joints'][0,0,i]}")
    print("=" * 50)

    # enable_print()
    return outputs

def run_mano_left(trans, root_orient, hand_pose, is_right=None, betas=None, use_cuda=True, fix_shapedirs=True):
    """
    Forward pass of the SMPL model and populates pred_data accordingly with
    joints3d, verts3d, points3d.

    trans : B x T x 3
    root_orient : B x T x 3
    body_pose : B x T x J*3
    betas : (optional) B x D
    """
    block_print()
    MANO_cfg = {
        'DATA_DIR': '_DATA/data_left/',
        'MODEL_PATH': '_DATA/data_left/mano_left',
        'GENDER': 'neutral',
        'NUM_HAND_JOINTS': 15,
        'CREATE_BODY_POSE': False,
        'is_rhand': False
    }
    mano_cfg = {k.lower(): v for k,v in MANO_cfg.items()}
    mano = MANO(**mano_cfg)
    if use_cuda:
        mano = mano.cuda()
    
    # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
    if fix_shapedirs:
        mano.shapedirs[:, 0, :] *= -1

    # skinning_weights = mano.lbs_weights
    # print(f"skinning_weights:{skinning_weights}")
    # if skinning_weights is not None:
    #     understand_skinning_weights(skinning_weights)
    # else:
    #     print("Warning: 无法访问skinning weights，跳过分析")

    B, T, _ = root_orient.shape
    NUM_JOINTS = 15
    mano_params = {
        'global_orient': root_orient.reshape(B*T, -1),
        'hand_pose': hand_pose.reshape(B*T*NUM_JOINTS, 3),
        'betas': betas.reshape(B*T, -1),
    }
    rotmat_mano_params = mano_params
    rotmat_mano_params['global_orient'] = aa_to_rotmat(mano_params['global_orient']).view(B*T, 1, 3, 3)
    rotmat_mano_params['hand_pose'] = aa_to_rotmat(mano_params['hand_pose']).view(B*T, NUM_JOINTS, 3, 3)
    rotmat_mano_params['transl'] = trans.reshape(B*T, 3)

    if use_cuda:
        mano_output = mano(**{k: v.float().cuda() for k,v in rotmat_mano_params.items()}, pose2rot=False)
    else:
        mano_output = mano(**{k: v.float() for k,v in rotmat_mano_params.items()}, pose2rot=False)

    faces_right = mano.faces
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
    faces_right = np.concatenate([faces_right, faces_new], axis=0)
    faces_n = len(faces_right)
    faces_left = faces_right[:,[0,2,1]]
    
    outputs = {
        "joints": mano_output.joints.reshape(B, T, -1, 3),
        "vertices": mano_output.vertices.reshape(B, T, -1, 3),
    }

    if not is_right is None:
        # outputs["vertices"][..., 0] = (2*is_right-1)*outputs["vertices"][..., 0]
        # outputs["joints"][..., 0] = (2*is_right-1)*outputs["joints"][..., 0]
        is_right = (is_right[:, :, 0].cpu().numpy() > 0)
        faces_result = np.zeros((B, T, faces_n, 3))
        faces_right_expanded = np.expand_dims(np.expand_dims(faces_right, axis=0), axis=0) 
        faces_left_expanded = np.expand_dims(np.expand_dims(faces_left, axis=0), axis=0) 
        faces_result = np.where(is_right[..., np.newaxis, np.newaxis], faces_right_expanded, faces_left_expanded)
        outputs["faces"] = torch.from_numpy(faces_result.astype(np.int32))


    enable_print()
    return outputs

def run_mano_twohands(init_trans, init_rot, init_hand_pose, is_right, init_betas, use_cuda=True, fix_shapedirs=True):
    outputs_left = run_mano_left(init_trans[0:1], init_rot[0:1], init_hand_pose[0:1], None, init_betas[0:1], use_cuda=use_cuda, fix_shapedirs=fix_shapedirs)
    outputs_right = run_mano(init_trans[1:2], init_rot[1:2], init_hand_pose[1:2], None, init_betas[1:2], use_cuda=use_cuda)
    outputs_two = {
        "vertices": torch.cat((outputs_left["vertices"], outputs_right["vertices"]), dim=0),
        "joints": torch.cat((outputs_left["joints"], outputs_right["joints"]), dim=0)

    }
    return outputs_two

def understand_skinning_weights(skinning_weights):
    """
    理解skinning weights的结构和含义
    
    Args:
        skinning_weights: 蒙皮权重张量 [V x J]
                         V = 778 (顶点数)
                         J = 16 (关节数，包括根关节)
    """
    print("=== Skinning Weights 分析 ===")
    print(f"Shape: {skinning_weights.shape}")
    print(f"数据类型: {skinning_weights.dtype}")
    print(f"取值范围: [{skinning_weights.min():.6f}, {skinning_weights.max():.6f}]")
    
    # 检查每个顶点的权重是否归一化
    weight_sums = skinning_weights.sum(dim=1)
    print(f"权重和的范围: [{weight_sums.min():.6f}, {weight_sums.max():.6f}]")
    print(f"权重和是否接近1: {torch.allclose(weight_sums, torch.ones_like(weight_sums))}")
    
    # 分析每个关节影响的顶点数
    print("\n=== 每个关节的影响分析 ===")
    joint_names = [
        'root', 'pinky_mcp', 'pinky_pip', 'pinky_dip' ,
        'index_mcp', 'index_pip', 'index_dip',
        'middle_mcp', 'middle_pip', 'middle_dip',
        'ring_mcp', 'ring_pip', 'ring_dip',
        'thumb_mcp', 'thumb_pip', 'thumb_dip',
    ]
    
    for i, joint_name in enumerate(joint_names):
        if i < skinning_weights.shape[1]:
            # 获取该关节权重最高的前20个顶点
            joint_weights = skinning_weights[:, i]
            top_weights, top_vertices = torch.topk(joint_weights, min(50, len(joint_weights)))
            print(f"{joint_name}: top_vertices:{top_vertices}")
            
