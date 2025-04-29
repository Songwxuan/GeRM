import numpy as np

path = '/zhaohan/robo_lang/quadruped_rt1/sim_quadruped_data_info/go_to_texture_vase/000000/dict.npy'

print(np.load(path, allow_pickle=True).item())