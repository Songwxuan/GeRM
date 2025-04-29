from pytorch_robotics_transformer.transformer_network import TransformerNetwork as RT1
import gym
import torch
import numpy as np
from model_analysis import model_analysis
from pytorch_robotics_transformer.universal_sentence_encoder_large_5_onnx import load_onnx_model
from quadruped_rt1 import ROOT_PATH

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

BATCH_SIZE = 2
TIME_SEQUENCE_LENGTH = 6
HEIGHT = 256
WIDTH = 320
NUM_IMAGE_TOKENS = 2
EMBEDDING_DIM = 512

model = RT1(
    input_tensor_space = gym.spaces.Dict(
        {
            'image': gym.spaces.Box(low=0.0, high=1.0, 
                            shape=(TIME_SEQUENCE_LENGTH, HEIGHT, WIDTH), dtype=np.float32),
            'natural_language_embedding': gym.spaces.Box(low=-np.inf, high=np.inf, 
                            shape=[EMBEDDING_DIM], dtype=np.float32)
        }
    ), # observation space like dict. keys are image, natural_language_embedding
    output_tensor_space = gym.spaces.Dict(
        {
            'world_vector': gym.spaces.Box(low= -1.0, high= 1.0, shape=(3,), dtype=np.float32),
            'rotation_delta': gym.spaces.Box(low= -np.pi / 2  , high= np.pi / 2, shape=(3,), dtype=np.float32),
            'gripper_closedness_action': gym.spaces.Box(low= -1.0  , high= 1.0, shape=(1,), dtype=np.float32),
            'terminate_episode': gym.spaces.Discrete(2)
        }
    ), # action space like dict. keys are world_vector, rotation_delta, gripper_closedness_action, terminate_episode
    train_step_counter = 0,
    vocab_size = 256, # Dimensionality of tokens from the output layer. This is also dimensionality of tokens from the input layer.
    token_embedding_size = EMBEDDING_DIM, # RT1ImageTokenizer outputs(=context_image_tokens) has embedding dimension of token_embedding_size. This will finally be utilized in 1x1 Conv in EfficientNetEncoder class.
    num_layers = 8,
    layer_size = 256, # This corresponds to key_dim which is the size of each attention head for query, key and values.
    num_heads = 6,
    feed_forward_size = 256, # This corresponds to d_model which is embedding dimension of each token in transformer part.
    dropout_rate = 0.1,
    time_sequence_length = TIME_SEQUENCE_LENGTH,
    crop_size = 236,
    # action_order: Optional[List[str]] = None,
    use_token_learner = True,
    return_attention_scores = False,
    device=device)

# model_analysis(model)
sentence_encoder = load_onnx_model(ROOT_PATH + "pytorch_robotics_transformer/universal_sentence_encoder_large_5_onnx/model.onnx",
                                   use_cuda=use_cuda)

image = torch.randn(BATCH_SIZE, TIME_SEQUENCE_LENGTH, 3, HEIGHT, WIDTH, device=device)
instructions = [
    'bring me that apple sitting on the table',
    'please pass the butter'
]
assert len(instructions) == BATCH_SIZE
sentence_embedding = sentence_encoder.run(output_names=["outputs"], input_feed={"inputs": instructions})[0]
sentence_embedding = torch.from_numpy(sentence_embedding).unsqueeze(1).repeat(1, TIME_SEQUENCE_LENGTH, 1).to(device)
actions = {
    'world_vector': torch.zeros(BATCH_SIZE, TIME_SEQUENCE_LENGTH, 3, dtype=torch.float32, device=device),
    'rotation_delta': torch.zeros(BATCH_SIZE, TIME_SEQUENCE_LENGTH, 3, dtype=torch.float32, device=device),
    'gripper_closedness_action': torch.zeros(BATCH_SIZE, TIME_SEQUENCE_LENGTH, 1, dtype=torch.float32, device=device),
    'terminate_episode': torch.zeros(BATCH_SIZE, TIME_SEQUENCE_LENGTH, dtype=torch.int64, device=device)
}

observation = {
    'image': image,
    'natural_language_embedding': sentence_embedding
}
network_state = {
                'context_image_tokens': torch.randn(BATCH_SIZE, TIME_SEQUENCE_LENGTH, 8, EMBEDDING_DIM, device=device),
                'action_tokens': torch.randn(BATCH_SIZE, TIME_SEQUENCE_LENGTH, 8, device=device),
                # Stores where in the window we are.
                # This value is within range [0, time_sequence_length + 1].
                # When seq_idx == time_sequence_length, context_image_tokens and
                # action_tokens need to be shifted to the left.
                'seq_idx': torch.randint(0, TIME_SEQUENCE_LENGTH + 1, (BATCH_SIZE, 1), dtype=torch.int64, device=device)
                # Our data is like context_image_tokens + action_tokens + context_image_tokens + action_tokens + context_image_tokens ...
                # 1 time step means [context_image_tokens + action_tokens]
                # seq_idx means which time steps we are. But it is adjusted to time_sequence_length when it exceeds time_sequence_length.
                }

model.set_actions(actions)
print(model._actions)
output_actions, network_state = model(observation, network_state)
print(model._aux_info)