import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import torchvision.transforms as transforms
import os
import numpy as np
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


resize = 300
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

transform_train = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

transform_to_tensor = transforms.ToTensor()

class RT1Dataset_fail(Dataset):
    def __init__(self,
                 traj_path,
                 sample_dict,
                 dataset_type='sim',
                 time_sequence_length=6,
                 transform=transform_train):

        self.traj_path = traj_path
        self.transform = transform
        self.sample_dict = sample_dict
        self.time_sequence_length = time_sequence_length
        self.dataset_type = dataset_type

 
    def __getitem__(self, index):
        if len(self.sample_dict) < self.time_sequence_length:
            index = 0
        else:
            index = int(np.clip(index, 0, len(self.sample_dict) - self.time_sequence_length))
        images = []
        commands = []
        terminates = []

        idx = []
        # proprioceptions = []
        for i in range(self.time_sequence_length):
            current_idx = int(np.clip(index + i, 0, len(self.sample_dict) - 1))
            try:
                item = self.sample_dict[str(current_idx)]
            except:
                print('error')
                print(str(current_idx))
                print(self.sample_dict.keys())
                print(self.traj_path)
            # item = json.loads(self.sample_dict[current_idx])
            image = Image.open(item['image'])                      
            if self.transform is not None:
                image = self.transform(image)
            else:
                image = transform_to_tensor(image)
            list_data = [float(x) for x in item['command'].split()]
            command = torch.tensor(list_data, dtype=torch.float32)
            # print("fail_command的内容是!", command)
            terminate = torch.tensor(int(item['terminate']), dtype=torch.int64)
            # proprioception = torch.tensor(item['proprioception'], dtype=torch.float32)
            images.append(image)
            commands.append(command)
            terminates.append(terminate)
            idx.append(len(self.sample_dict) - current_idx)
            # proprioceptions.append(proprioception)
        images = torch.stack(images)
        # instructions = random.sample(item['instruction'], 1)[0]
        instructions = item['instruction']
        commands = torch.stack(commands)
        terminates = torch.stack(terminates)
        idx = torch.tensor(idx)
        # proprioceptions = torch.stack(proprioceptions)
        masks = torch.tensor([self.dataset_type == 'sim'], dtype=torch.float).unsqueeze(0).repeat(self.time_sequence_length, 1)
        is_success = 0
        return images, instructions, commands, terminates, masks, idx, is_success
 
    def __len__(self):
        return len(self.sample_dict)
    

class StateCommandDataset(Dataset):
    def __init__(self, 
                 dataset_type,
                 dataset_path):
        """
        dataset_type: ['train', 'test']
        """
 
        self.sample_dict = list()
        self.dataset_type = dataset_type
        f = open(dataset_path + '/datalist.json')
        lines = f.readlines()
        for line in lines:
            self.sample_dict.append(line.strip())
        f.close()
 
    def __getitem__(self, index):
        item = json.loads(self.sample_dict[index])
        commands = torch.tensor(item['command'], dtype=torch.float32)
        proprioceptions = torch.tensor(item['proprioception'], dtype=torch.float32)
        return commands, proprioceptions
 
    def __len__(self):

        return len(self.sample_dict)

if __name__ == '__main__':
    import time

    sim_file = open('dataset_info/sim_quadruped_data_info.json', 'r')
    sim_dict = json.load(sim_file)
    count = 0
    for key in sim_dict.keys():
        sim_dataset = RT1Dataset_fail(sim_dict[key])
        count += 1
        if count > 0:
            break
    print(len(sim_dataset))
    print(sim_dict[key])
