from pytorch_robotics_transformer.transformer_network import TransformerNetwork as RT1
import gym
import torch
import numpy as np
from model_analysis import model_analysis
from pytorch_robotics_transformer.universal_sentence_encoder_large_5_onnx import load_onnx_model
from __init__ import ROOT_PATH

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

batch_size = 2
time_sequence_length = 6
image_height = 256
image_width = 320
NUM_IMAGE_TOKENS = 2
embedding_dim = 512
action_dim = 11
action_space_low = np.array([-1.] * action_dim)
action_space_high = np.array([1.] * action_dim)
state_dim = 31

model = RT1(input_tensor_space = gym.spaces.Dict(
                {
                    'image': gym.spaces.Box(low=0.0, high=1.0, 
                                    shape=(1, image_height, image_width), dtype=np.float32),
                    'natural_language_embedding': gym.spaces.Box(low=-np.inf, high=np.inf, 
                                    shape=[embedding_dim], dtype=np.float32),
                    'proprioception': gym.spaces.Box(low=-np.inf, high=np.inf,
                                    shape=[state_dim], dtype=np.float32)
                }
            ), # observation space like dict. keys are image, natural_language_embedding
            output_tensor_space = gym.spaces.Dict(
                {   
                    'terminate_episode': gym.spaces.Discrete(2),
                    'commands': gym.spaces.Box(low=action_space_low, high=action_space_high, dtype=np.float32)
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
            use_action_quantizer = False,
            return_attention_scores = False,
            device=device,
        )

image = torch.randn(batch_size, time_sequence_length, 3, image_height, image_width, device=device)
sentence_embedding = torch.randn(batch_size, time_sequence_length, embedding_dim, device=device)
actions = {
    'terminate_episode': torch.zeros(batch_size, time_sequence_length, dtype=torch.int64, device=device),
    'commands': torch.randn(batch_size, time_sequence_length, action_dim, device=device)
}

observation = {
    'image': image,
    'natural_language_embedding': sentence_embedding,
    'proprioception': torch.randn(batch_size, time_sequence_length, state_dim, device=device)
}
network_state = {
                'context_image_tokens': torch.randn(batch_size, time_sequence_length, 8, embedding_dim, device=device),
                'action_tokens': torch.randn(batch_size, time_sequence_length, 8, device=device),
                # Stores where in the window we are.
                # This value is within range [0, time_sequence_length + 1].
                # When seq_idx == time_sequence_length, context_image_tokens and
                # action_tokens need to be shifted to the left.
                'seq_idx': torch.randint(0, time_sequence_length + 1, (batch_size, 1), dtype=torch.int64, device=device)
                # Our data is like context_image_tokens + action_tokens + context_image_tokens + action_tokens + context_image_tokens ...
                # 1 time step means [context_image_tokens + action_tokens]
                # seq_idx means which time steps we are. But it is adjusted to time_sequence_length when it exceeds time_sequence_length.
                }
# model._action_tokenizer.eval()
model.set_actions(actions)
for key in model._actions:
    print(key, model._actions[key].shape)
output_actions, network_state = model(observation, network_state)
for key in model._aux_info:
    print(key, model._aux_info[key].shape)