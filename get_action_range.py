import numpy as np
import os
from tqdm import tqdm
from __init__ import ROOT_PATH
from quadruped_rt1.dataset_old import INSTRUCTION_DICT

data_path = os.path.join(ROOT_PATH, 'sim_quadruped_data')
info_path = os.path.join(ROOT_PATH, 'sim_quadruped_data_info')
tasks = next(os.walk(data_path))[1]
if not os.path.exists(info_path):
    os.mkdir(info_path)

dx_max, dx_min = -100, 100
dy_max, dy_min = -100, 100
dyaw_max, dyaw_min = -100, 100
body_height_max, body_height_min = -100, 100
step_frequency_max, step_frequency_min = -100, 100
gait_max, gait_min = 1.0, 0.0
footswing_height_max, footswing_height_min = -100, 100
pitch_max, pitch_min = -100, 100
roll_max, roll_min = -100, 100
stance_width_max, stance_width_min = -100, 100
proprio_mean, proprio_std = np.zeros(31), np.zeros(31)

gravity_vec = np.array([0, 0, -1.])
def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.sum(q_vec * v) * 2.0
    return a - b + c

count_data = 0
for task in tqdm(tasks):
    if task in list(INSTRUCTION_DICT.keys()):
        task_path = os.path.join(data_path, task)
        task_info_path = os.path.join(info_path, task)
        if not os.path.exists(task_info_path):
            os.mkdir(task_info_path)
        episode_files = os.listdir(task_path)
        
        num_episodes = len(episode_files)
        print("task: {}".format(task, num_episodes))
        
        for episode in episode_files:
            episode_path = os.path.join(task_path, episode)
            
            episode_info_path = os.path.join(task_info_path, episode)
                
            episode_command_path = os.path.join(episode_path, "command")
            episode_action_path = os.path.join(episode_path, "action")

            if os.path.exists(episode_command_path):
                commands = os.listdir(episode_command_path)
                episode_length = len(commands)
                if episode_length > 50:
                    if not os.path.exists(episode_info_path):
                        os.mkdir(episode_info_path)
                
                    dict = {}
                    dx_dict = {}
                    dy_dict = {}
                    dyaw_dict = {}
                    body_height_dict = {}
                    step_frequency_dict = {}
                    gait_0_dict = {}
                    gait_1_dict = {}
                    gait_2_dict = {}
                    footswing_height_dict = {}
                    pitch_dict = {}
                    roll_dict = {}
                    stance_width_dict = {}
                    proprioception_dict = {}
                    
                    
                    for i in range(0, episode_length, 10):
                        # command_data_path = os.path.join(episode_command_path, command)
                        command_data_path = os.path.join(episode_command_path, '{:03d}.npy'.format(i))
                        action_data_path = os.path.join(episode_action_path, '{:03d}.npy'.format(i))
                        if os.path.getsize(command_data_path) == 0:
                            command_data_path = os.path.join(episode_command_path, '{:03d}.npy'.format(i - 1))
                        # print(command_data_path)
                        data = np.load(command_data_path, allow_pickle=True).item()
                        proprio_data = np.load(action_data_path, allow_pickle=True).item()
                        
                        dx = float(data['x_vel_cmd'])
                        dy = float(data['y_vel_cmd'])
                        dyaw = float(data['yaw_vel_cmd'])
                        body_height = float(data['body_height_cmd'])
                        step_frequency = float(data['step_frequency_cmd'])
                        gait_0 = float(data['gait '][0])
                        gait_1 = float(data['gait '][1])
                        gait_2 = float(data['gait '][2])
                        footswing_height = float(data['footswing_height_cmd'])
                        pitch = float(data['pitch_cmd'])
                        roll = float(data['roll_cmd'])
                        stance_width = float(data['stance_width_cmd'])
                        
                        body_quat = proprio_data['body_quat']
                        projected_gravity = quat_rotate_inverse(body_quat, gravity_vec)
                        joint_pos = proprio_data['joint_pos']
                        joint_vel = proprio_data['joint_vel']
                        contact_states = proprio_data['contact_states']
                        proprioception = np.concatenate((projected_gravity, joint_pos, joint_vel, contact_states), axis=0)

                        
                        dx_dict[i] = dx 
                        dy_dict[i] = dy
                        dyaw_dict[i] = dyaw
                        body_height_dict[i] = body_height
                        step_frequency_dict[i] = step_frequency
                        gait_0_dict[i] = gait_0
                        gait_1_dict[i] = gait_1
                        gait_2_dict[i] = gait_2
                        footswing_height_dict[i] = footswing_height
                        pitch_dict[i] = pitch
                        roll_dict[i] = roll
                        stance_width_dict[i] = stance_width

                        proprioception_dict[i] = proprioception
                        
                        if dx > dx_max:
                            dx_max = dx
                        if dy > dy_max:
                            dy_max = dy
                        if dyaw > dyaw_max:
                            dyaw_max = dyaw
                        if body_height > body_height_max:
                            body_height_max = body_height
                        if step_frequency > step_frequency_max:
                            step_frequency_max = step_frequency
                        if footswing_height > footswing_height_max:
                            footswing_height_max = footswing_height
                        if pitch > pitch_max:
                            pitch_max = pitch
                        if roll > roll_max:
                            roll_max = roll
                        if stance_width > stance_width_max:
                            stance_width_max = stance_width
                        if dx < dx_min:
                            dx_min = dx
                        if dy < dy_min:
                            dy_min = dy
                        if dyaw < dyaw_min:
                            dyaw_min = dyaw
                        if body_height < body_height_min:
                            body_height_min = body_height
                        if step_frequency < step_frequency_min:
                            step_frequency_min = step_frequency
                        if footswing_height < footswing_height_min:
                            footswing_height_min = footswing_height
                        if pitch < pitch_min:
                            pitch_min = pitch
                        if roll < roll_min:
                            roll_min = roll
                        if stance_width < stance_width_min:
                            stance_width_min = stance_width

                        # running mean and std
                        if count_data == 0:
                            proprio_mean = proprioception
                            proprio_std = np.zeros(31)
                        else:
                            proprio_mean = (proprio_mean * count_data + proprioception) / (count_data + 1)
                            proprio_std = (proprio_std * count_data + (proprioception - proprio_mean) ** 2) / (count_data + 1)

                        count_data += 1
                            
                        dict['dx'] = dx_dict
                        dict['dy'] = dy_dict
                        dict['dyaw'] = dyaw_dict
                        dict['body_height'] = body_height_dict
                        dict['step_frequency'] = step_frequency_dict
                        dict['gait_0'] = gait_0_dict
                        dict['gait_1'] = gait_1_dict
                        dict['gait_2'] = gait_2_dict
                        dict['footswing_height'] = footswing_height_dict
                        dict['pitch'] = pitch_dict
                        dict['roll'] = roll_dict
                        dict['stance_width'] = stance_width_dict
                        dict['proprioception'] = proprioception_dict
                                
                    
                    dict_path = os.path.join(episode_info_path, "dict.npy")
                    np.save(dict_path, dict)

dx_range = dx_max - dx_min
dy_range = dy_max - dy_min
dyaw_range = dyaw_max - dyaw_min
body_height_range = body_height_max - body_height_min
step_frequency_range = step_frequency_max - step_frequency_min
footswing_height_range = footswing_height_max - footswing_height_min
pitch_range = pitch_max - pitch_min
roll_range = roll_max - roll_min
stance_width_range = stance_width_max - stance_width_min

command_space_low = np.array([dx_min, dy_min, dyaw_min, body_height_min, step_frequency_min, gait_min, gait_min, gait_min, footswing_height_min, pitch_min, stance_width_min])
command_space_high = np.array([dx_max, dy_max, dyaw_max, body_height_max, step_frequency_max, gait_max, gait_max, gait_max, footswing_height_max, pitch_max, stance_width_max])

range_dict = {}
range_dict['dx_range'] = dx_range
range_dict['dy_range'] = dy_range
range_dict['dyaw_range'] = dyaw_range
range_dict['body_height_range'] = body_height_range
range_dict['step_frequency_range'] = step_frequency_range
range_dict['footswing_height_range'] = footswing_height_range
range_dict['pitch_range'] = pitch_range
range_dict['roll_range'] = roll_range
range_dict['stance_width_range'] = stance_width_range
range_dict['dx_max'] = dx_max
range_dict['dx_min'] = dx_min
range_dict['dy_max'] = dy_max
range_dict['dy_min'] = dy_min
range_dict['dyaw_max'] = dyaw_max
range_dict['dyaw_min'] = dyaw_min
range_dict['body_height_max'] = body_height_max
range_dict['body_height_min'] = body_height_min
range_dict['step_frequency_max'] = step_frequency_max
range_dict['step_frequency_min'] = step_frequency_min
range_dict['footswing_height_max'] = footswing_height_max
range_dict['footswing_height_min'] = footswing_height_min
range_dict['pitch_max'] = pitch_max
range_dict['pitch_min'] = pitch_min
range_dict['roll_max'] = roll_max
range_dict['roll_min'] = roll_min
range_dict['stance_width_max'] = stance_width_max
range_dict['stance_width_min'] = stance_width_min
range_dict['command_space_low'] = command_space_low
range_dict['command_space_high'] = command_space_high
range_dict['proprio_mean'] = proprio_mean
range_dict['proprio_std'] = proprio_std
range_dict['count_data'] = count_data

print("command_space_low:", command_space_low)
print("command_space_high:", command_space_high)
range_dict_path = os.path.join(info_path, "ranges.npy")
np.save(range_dict_path, range_dict)
