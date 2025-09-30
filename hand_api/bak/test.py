from rh20t_api.configurations import load_conf
from rh20t_api.scene import RH20TScene

scene_path = "/share/zsp/datasets/motion_raw/hoi/RH20T/data/RH20T_cfg1/task_0215_user_0015_scene_0010_cfg_0001"
robot_configs = load_conf("configs/configs.json")
scene = RH20TScene(scene_path, robot_configs)
# RH20TScene.get_tcp_aligned(timestamp:int, serial:str="base")

print("extrinsics_base_aligned", scene.extrinsics_base_aligned)
print("folder", scene.folder)
print("intrinsics", scene.intrinsics)
print("in_hand_serials", scene.in_hand_serials)
print("serials", scene.serials)
print("start_timestamp", scene.start_timestamp)
print("end_timestamp", scene.end_timestamp)
print("get_image_path_pairs", scene.get_image_path_pairs(scene.start_timestamp, image_types=["color"]))
print("get_tcp_aligned", scene.get_tcp_aligned(scene.start_timestamp, scene.serials[0]))
print("get_tcp_aligned", scene.get_tcp_aligned(scene.start_timestamp, scene.serials[1]))
print("get_tcp_aligned", scene.get_tcp_aligned(scene.start_timestamp, scene.serials[2]))
print("get_gripper_command", scene.get_gripper_command(scene.start_timestamp+150))
print("get_gripper_info", scene.get_gripper_info(scene.start_timestamp+150))


# 命名规则
# /share/zsp/datasets/motiondb/motionlib-hoi/RH20T_robot
# - feature - tcp_gripper - task_0215 - task_0215_user_0015_scene_0010_cfg_0001_serial.npy
# - images - images - task_0215 - task_0215_user_0015_scene_0010_cfg_0001_serial - jpg1, jpg2
# - text - task_description - task_0215 - task_0215_user_0015_scene_0010_cfg_0001_serial.txt

# grippers = []
# serials = scene.serials
# for i in range(scene.start_timestamp, scene.end_timestamp):
#     grippers.append(scene.get_gripper_command(i))
#     # print()
# print(len(grippers))
# print(grippers[10000:10100])
# # print(scene.get_tcp_aligned(timestamp:int, serial:str="base"))