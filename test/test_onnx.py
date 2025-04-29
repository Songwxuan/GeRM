import torch
import onnx
import gym
import numpy as np
import time
from copy import deepcopy
from pytorch_robotics_transformer.transformer_network import TransformerNetwork as RT1

image_height, image_width = 300, 300
embedding_dim = 512
time_sequence_length = 6
action_dim = 11
action_space_low = np.zeros(11)
action_space_high = np.ones(11)
batch_size = 16
device = "cuda:0"

model = RT1(input_tensor_space = gym.spaces.Dict(
                {
                    'image': gym.spaces.Box(low=0.0, high=1.0, 
                                    shape=(1, image_height, image_width), dtype=np.float32),
                    'natural_language_embedding': gym.spaces.Box(low=-np.inf, high=np.inf, 
                                    shape=[embedding_dim], dtype=np.float32)
                }
            ), # observation space like dict. keys are image, natural_language_embedding
            output_tensor_space = gym.spaces.Dict(
                {
                    'commands': gym.spaces.Box(low=action_space_low, high=action_space_high, dtype=np.float32),
                    'terminate_episode': gym.spaces.Discrete(2)
                }
            ), # action space like dict. keys are commands, terminate_episode
            train_step_counter = 0,
            vocab_size = 256, # Dimensionality of tokens from the output layer. This is also dimensionality of tokens from the input layer.
            token_embedding_size = embedding_dim, # RT1ImageTokenizer outputs(=context_image_tokens) has embedding dimension of token_embedding_size. This will finally be utilized in 1x1 Conv in EfficientNetEncoder class.
            num_layers = 8,
            layer_size = 256, # This corresponds to key_dim which is the size of each attention head for query, key and values.
            num_heads = 6,
            feed_forward_size = 256, # This corresponds to d_model which is embedding dimension of each token in transformer part.
            dropout_rate = 0.1,
            time_sequence_length = time_sequence_length,
            # action_order: Optional[List[str]] = None,
            use_token_learner = True,
            return_attention_scores = False,
            device=device,
            use_cache=False
        )

model.eval()

observation = {
            'image': torch.randn((batch_size, 3, image_height, image_width), device=device),
            'natural_language_embedding': torch.randn((batch_size, embedding_dim), device=device)
        }

ori_network_state = {
                'context_image_tokens': torch.randint(0, 256, (batch_size, time_sequence_length, 8, embedding_dim), device=device),
                'action_tokens': torch.randint(0, 256, (batch_size, time_sequence_length, action_dim + 1), device=device),
                # Stores where in the window we are.
                # This value is within range [0, time_sequence_length + 1].
                # When seq_idx == time_sequence_length, context_image_tokens and
                # action_tokens need to be shifted to the left.
                'seq_idx': torch.zeros((1, 1), dtype=torch.int64, device=device)
                # Our data is like context_image_tokens + action_tokens + context_image_tokens + action_tokens + context_image_tokens ...
                # 1 time step means [context_image_tokens + action_tokens]
                # seq_idx means which time steps we are. But it is adjusted to time_sequence_length when it exceeds time_sequence_length.
                }

network_state = deepcopy(ori_network_state)

time_0 = time.time()
for _ in range(10):
    with torch.no_grad():
        _, network_state = model(observation, network_state)
time_1 = time.time()
print("Pytorch:", time_1 - time_0)
input_names = ["observation", "input_network_state"]
output_names = ["output_actions", 'output_network_state']

# network_state = deepcopy(ori_network_state)

# # torch.onnx.export(model,(observation, network_state),'model.onnx',input_names=input_names,output_names=output_names,
# #   dynamic_axes={'observation':[0],'input_network_state':[0], 'output_actions':[0], 'output_network_state':[0]} )

# script_model = torch.jit.script(model)


# time_0 = time.time()
# for _ in range(10):
#     with torch.no_grad():
#         _, network_state = script_model(observation, network_state)
# time_1 = time.time()
# print("Onnx:", time_1 - time_0)

