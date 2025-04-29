import torch
import os
import numpy as np
import json
from tqdm import tqdm

real_dict_path = '/wangdonglin/quadruped_data_with_comand_info/all_dict.npy'
sim_dict_path = '/wangdonglin/sim_quadruped_data_info/all_dict.npy'

real_dict = np.load(real_dict_path, allow_pickle=True).item()
sim_dict = np.load(sim_dict_path, allow_pickle=True).item()


f = open('dataset_info/sim_quadruped_data_info.json', 'w')
info_dict = {}
for key in tqdm(sim_dict.keys()):
    dict = sim_dict[key]
    episode_length = len(dict['dx'])
    key_dict = {key:{}}
    for i in range(episode_length):
        true_idx = i * 10
        img_idx = "{:03d}".format(true_idx)
        img_path = os.path.join(f'{key}/image', f"{img_idx}.png")

        dx = dict['dx'][true_idx]
        dy = dict['dy'][true_idx]
        dyaw = dict['dyaw'][true_idx]
        body_height = dict['body_height'][true_idx]
        step_frequency = dict['step_frequency'][true_idx]
        gait_0 = dict['gait_0'][true_idx]
        gait_1 = dict['gait_1'][true_idx]
        gait_2 = dict['gait_2'][true_idx]
        footswing_height = dict['footswing_height'][true_idx]
        pitch = dict['pitch'][true_idx]
        stance_width = dict['stance_width'][true_idx]

        instruction = dict['instruction']
        terminate = int(i == episode_length - 1)
        command = np.array([dx, dy, dyaw, body_height, step_frequency, gait_0, gait_1, gait_2, footswing_height, pitch, stance_width])

        step_dict = {i: {
            'image': img_path,
            'instruction': instruction,
            'command': command.tolist(),
            'terminate': terminate
        }}

        key_dict[key].update(step_dict)
    info_dict.update(key_dict)
json.dump(info_dict, f)
f.close()

f = open('dataset_info/quadruped_data_info.json', 'w')
info_dict = {}
for key in tqdm(real_dict.keys()):
    dict = real_dict[key]
    episode_length = len(dict['dx'])
    key_dict = {key:{}}
    for i in range(episode_length):
        true_idx = i * 2
        img_idx = true_idx
        img_path = os.path.join(f'{key}/image', f"{img_idx}.jpg")

        dx = dict['dx'][true_idx]
        dy = dict['dy'][true_idx]
        dyaw = dict['dyaw'][true_idx]
        body_height = dict['body_height'][true_idx]
        step_frequency = dict['step_frequency'][true_idx]
        gait_0 = dict['gait_0'][true_idx]
        gait_1 = dict['gait_1'][true_idx]
        gait_2 = dict['gait_2'][true_idx]
        footswing_height = dict['footswing_height'][true_idx]
        pitch = dict['pitch'][true_idx]
        stance_width = dict['stance_width'][true_idx]

        instruction = dict['instruction']
        terminate = int(i == episode_length - 1)
        command = np.array([dx, dy, dyaw, body_height, step_frequency, gait_0, gait_1, gait_2, footswing_height, pitch, stance_width])

        step_dict = {i: {
            'image': img_path,
            'instruction': instruction,
            'command': command.tolist(),
            'terminate': terminate
        }}

        key_dict[key].update(step_dict)
    info_dict.update(key_dict)
json.dump(info_dict, f)
f.close()