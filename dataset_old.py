import torch
import yaml
from torch.utils.data import Dataset
from PIL import Image
import json
import copy
import torchvision.transforms as transforms
import pandas as pd
import random
import os
import numpy as np
import json
import pandas as pd
from __init__ import ROOT_PATH

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# create data
# transform = transforms.Compose([
#     transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC,
#                                  antialias=None),  # 3 is bicubic
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=(300, 300), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC,
                                 antialias=None),  # 3 is bicubic
    transforms.ToTensor(),
    ])

transform_to_tensor = transforms.ToTensor()

INSTRUCTION_DICT = {
    'go_avoid_orange_cube': 'go to the orange cube and avoid the obstacle',
    'go_avoid_purple_ball': 'go to the purple ball and avoid the obstacle',
    'go_avoid_red_cube': 'go to the red cube and avoid the obstacle',
    'go_avoid_tan_ball': 'go to the tan ball and avoid the obstacle',
    'go_avoid_texture_barrel': 'go to the barrel and avoid the obstacle',
    'go_avoid_texture_bench': 'go to the bench and avoid the obstacle',
    'go_avoid_texture_bookshelf': 'go to the bookshelf and avoid the obstacle',
    'go_avoid_texture_chair': 'go to the chair and avoid the obstacle',
    'go_avoid_texture_cooker': 'go to the cooker and avoid the obstacle',
    'go_avoid_texture_drawers': 'go to the drawers and avoid the obstacle',
    'go_avoid_texture_fan': 'go to the fan and avoid the obstacle',
    'go_avoid_texture_fence': 'go to the fence and avoid the obstacle',
    'go_avoid_texture_oven': 'go to the oven and avoid the obstacle',
    'go_avoid_texture_piano': 'go to the piano and avoid the obstacle',
    'go_avoid_texture_sofa': 'go to the sofa and avoid the obstacle',
    'go_avoid_texture_table': 'go to the table and avoid the obstacle',
    'go_avoid_texture_trashcan': 'go to the trashcan and avoid the obstacle',
    'go_avoid_texture_vase': 'go to the vase and avoid the obstacle',
    'go_avoid_texture_wardrobe': 'go to the wardrobe and avoid the obstacle',
    'go_to_orange_ball': 'go to the orange ball',
    'go_to_purple_cube': 'go to the purple cube',
    'go_to_texture_barrel': 'go to the barrel',
    'go_to_texture_bench': 'go to the bench',
    'go_to_texture_bookshelf': 'go to the bookshelf',
    'go_to_texture_chair': 'go to the chair',
    'go_to_texture_cooker': 'go to the cooker',
    'go_to_texture_drawers': 'go to the drawers',
    'go_to_texture_fan': 'go to the fan',
    'go_to_texture_fence': 'go to the fence',
    'go_to_texture_oven': 'go to the oven',
    'go_to_texture_piano': 'go to the piano',
    'go_to_texture_sofa': 'go to the sofa',
    'go_to_texture_table': 'go to the table',
    'go_to_texture_trashcan': 'go to the trashcan',
    'go_to_texture_vase': 'go to the vase',
    'go_to_texture_wardrobe': 'go to the wardrobe',
    'stop_beige_ball': 'stop the beige ball',
    'stop_black_ball': 'stop the black ball',
    'stop_blue_ball': 'stop the blue ball',
    'stop_brown_ball': 'stop the brown ball',
    'stop_cyan_ball': 'stop the cyan ball',
    'stop_gold_ball': 'stop the gold ball',
    'stop_gray_ball': 'stop the gray ball',
    'stop_green_ball': 'stop the green ball',
    'stop_navy_ball': 'stop the navy ball',
    'stop_orange_ball': 'stop the orange ball',
    'stop_pink_ball': 'stop the pink ball',
    'stop_purple_ball': 'stop the purple ball',
    'stop_red_ball': 'stop the red ball',
    'stop_silver_ball': 'stop the silver ball',
    'stop_tan_ball': 'stop the tan ball',
    'stop_white_ball': 'stop the white ball',
    'stop_yellow_ball': 'stop the yellow ball',
}

def make_json_file(data_path, info_path, task):
    json_path = os.path.join(info_path, "datalist.json")
    dict_path = os.path.join(info_path, "dict.npy")
    if not os.path.exists(dict_path):
        print(f'No dict file in {info_path}')
        os.rmdir(info_path)
    else:
        if os.path.exists(json_path):
            os.remove(json_path)
        print(f'Writing to {json_path}')
        dict = np.load(dict_path, allow_pickle=True).item()
        episode_length = len(dict['dx'])
        f = open(json_path, 'w')
        for i in range(episode_length):
            true_idx = i * 10
            img_idx = "{:03d}".format(true_idx)
            img_path = os.path.join(data_path, f"image/{img_idx}.png")
            
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

            proprioception = dict['proprioception'][true_idx]
            
            terminate = int(i == episode_length - 1)

            command = np.array([dx, dy, dyaw, body_height, step_frequency, gait_0, gait_1, gait_2, footswing_height, pitch, stance_width])

            instruction = INSTRUCTION_DICT[task]
            csv_path = os.path.join(data_path, f"info.csv")
            df = pd.read_csv(csv_path, encoding="utf-8")
            gait = df['gait'][0]
            vel = df['instruction'][0].split(", ")[-1]
            specific_instruction1 = instruction + f' {vel}'
            specific_instruction2 = instruction + f' with a {gait} gait'
            specific_instruction3 = specific_instruction1 + f' with a {gait} gait'
            instruction_list = [instruction, specific_instruction1, specific_instruction2, specific_instruction3]
            # if vel != 'normally' or gait != 'trotting':
            #     instruction_list = instruction_list[1:]
            
            
            dict_ = {
                'image': img_path,
                'instruction': instruction_list,
                'command': command.tolist(),
                'terminate': terminate,
                'proprioception': proprioception.tolist()
            }

            json.dump(dict_, f)
            if i != episode_length - 1:
                f.write('\n')

        f.close()

 
class RT1Dataset(Dataset):
    def __init__(self, 
                 dataset_type,
                 dataset_path,
                 time_sequence_length=6,
                 transform=None):
        """
        dataset_type: ['train', 'test']
        """
 
        self.transform = transform
        self.sample_list = list()
        self.dataset_type = dataset_type
        self.dataset_path = dataset_path
        self.time_sequence_length = time_sequence_length
        f = open(dataset_path + '/datalist.json')
        lines = f.readlines()
        for line in lines:
            self.sample_list.append(line.strip())
        f.close()
 
    def __getitem__(self, index):
        if len(self.sample_list) < self.time_sequence_length:
            index = 0
        else:
            index = int(np.clip(index, 0, len(self.sample_list) - self.time_sequence_length))
        images = []
        commands = []
        terminates = []
        proprioceptions = []
        for i in range(self.time_sequence_length):
            current_idx = int(np.clip(index + i, 0, len(self.sample_list) - 1))
            item = json.loads(self.sample_list[current_idx])
            image = Image.open(item['image'])

            if self.transform is not None:
                image = self.transform(image)
            else:
                image = transform_to_tensor(image)
            command = torch.tensor(item['command'], dtype=torch.float32)
            terminate = torch.tensor(int(item['terminate']), dtype=torch.int64)
            proprioception = torch.tensor(item['proprioception'], dtype=torch.float32)
            images.append(image)
            commands.append(command)
            terminates.append(terminate)
            proprioceptions.append(proprioception)
        images = torch.stack(images)
        # instructions = random.sample(item['instruction'], 1)[0]
        instructions = item['instruction'][-1]
        commands = torch.stack(commands)
        terminates = torch.stack(terminates)
        proprioceptions = torch.stack(proprioceptions)
        return images, instructions, commands, terminates, proprioceptions, self.dataset_path
 
    def __len__(self):
        return len(self.sample_list)
    

class StateCommandDataset(Dataset):
    def __init__(self, 
                 dataset_type,
                 dataset_path):
        """
        dataset_type: ['train', 'test']
        """
 
        self.sample_list = list()
        self.dataset_type = dataset_type
        f = open(dataset_path + '/datalist.json')
        lines = f.readlines()
        for line in lines:
            self.sample_list.append(line.strip())
        f.close()
 
    def __getitem__(self, index):
        item = json.loads(self.sample_list[index])
        commands = torch.tensor(item['command'], dtype=torch.float32)
        proprioceptions = torch.tensor(item['proprioception'], dtype=torch.float32)
        return commands, proprioceptions
 
    def __len__(self):
        return len(self.sample_list)

if __name__ == '__main__':
    data_path = os.path.join(ROOT_PATH, 'sim_quadruped_data')
    info_path = os.path.join(ROOT_PATH, 'sim_quadruped_data_info')
    task_list = INSTRUCTION_DICT.keys()

    for task in task_list:
        task_path = os.path.join(data_path, task)
        task_info_path = os.path.join(info_path, task)
        if os.path.exists(task_info_path):
            episode_list = os.listdir(task_path)
            for episode in episode_list:
                episode_path = os.path.join(task_path, episode)
                episode_command_path = os.path.join(episode_path, "command")
                episode_info_path = os.path.join(task_info_path, episode)
                if os.path.exists(episode_info_path):
                    make_json_file(episode_path, episode_info_path, task)
