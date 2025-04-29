import numpy as np

action_path = '/zhaohan/robo_lang/quadruped_rt1/000006/action/000.npy'
command_path = '/zhaohan/robo_lang/quadruped_rt1/000006/command/000.npy'

action = np.load(action_path, allow_pickle=True).item()

command = np.load(command_path, allow_pickle=True).item()

print(action.keys())

print(command.keys())