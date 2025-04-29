import os
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=(300, 300), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC,
                                 antialias=None),  # 3 is bicubic
    transforms.ToTensor(),
    ])

file_path = "/zhaohan/robo_lang/quadruped_rt1/sim_quadruped_data/"

task_list = os.listdir(file_path)

# print(task_list)
for task in tqdm(task_list):
    task_path = file_path + task
    task_data = os.listdir(task_path)
    # print(task_data)
    for data in task_data:
        data_path = task_path + "/" + data + "/image"
        # print(data_path)
        if os.path.exists(data_path):
            data_file = os.listdir(data_path)
            if data_file == []:
                print(data_path)
            else:
                file = "{:03d}.png".format(len(data_file) - 1)
                data_file_path = data_path + "/" + file
                print(data_file_path)
                image = Image.open(data_file_path)
                try:
                    image = transform_train(image)
                except:
                    print(data_file_path)
        else:
            print(data_path)