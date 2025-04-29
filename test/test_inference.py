import torch
import os
from __init__ import ROOT_PATH
from agent import Agent

def reset():
    images = torch.randn(1, 3, 300, 300, device=device)
    instructions = ['bring me that apple sitting on the table']
    return images, instructions

def step(actions):
    return torch.randn(1, 3, 300, 300, device=device)

log_to_wandb = False
batch_size = 4
num_epoch = 10
time_sequence_length = 6

image_height = 300
image_width = 300
embedding_dim = 512
# action_dim = 11
use_path = os.path.join(ROOT_PATH, 'pytorch_robotics_transformer/universal_sentence_encoder_large_5_onnx/model.onnx')
#wx: Test
model_path = os.path.join(ROOT_PATH, 'outputs/2023-10-22-19-54-01/model_epoch_3.pth')
data_path = os.path.join(ROOT_PATH, 'sim_quadruped_data')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

agent = Agent(batch_size,
                time_sequence_length,
                image_height,
                image_width,
                embedding_dim,
                use_path,
                data_path,
                device,
                log_to_wandb,
                num_epoch)
agent.load(model_path)

network_state = None
images, instructions = reset()

for timestep in range(100):
    output_actions, network_state = agent.inference_one_time_step(images, instructions, network_state)
    images = step(output_actions)
    print(output_actions['commands'].shape)
    if output_actions['terminate_episode'][0] == 1:
        break