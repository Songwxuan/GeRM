import gym
import torch
import numpy as np
import pytz
import os
import wandb
from datetime import datetime
from dataset import RT1Dataset, transform_train, INSTRUCTION_DICT
from torch.utils.data import DataLoader
from model_analysis import model_analysis
from pytorch_robotics_transformer.transformer_network import TransformerNetwork as RT1
from pytorch_robotics_transformer.universal_sentence_encoder_large_5_onnx import load_onnx_model
from multiprocessing import cpu_count
from __init__ import ROOT_PATH

class Agent:
    def __init__(self,
                 batch_size: int=2,
                 time_sequence_length: int=6,
                 image_height: int=256,
                 image_width: int=320,
                 embedding_dim: int=512,
                 use_path: str=os.path.join(ROOT_PATH, 'pytorch_robotics_transformer/universal_sentence_encoder_large_5_onnx/model.onnx'),
                 data_path: str=os.path.join(ROOT_PATH, 'sim_quadruped_data'),
                 action_quantizer_path: str=os.path.join(ROOT_PATH, 'outputs/action_quantizer/action_quantizer.pth'),
                 device=torch.device("cpu"),
                 log_to_wandb: bool=False,
                 use_action_quantizer: bool=False,
                 num_epoch: int=10):
        self.time_sequence_length = time_sequence_length
        self.image_height = image_height
        self.image_width = image_width
        self.embedding_dim = embedding_dim
        self.device = device
        self.log_to_wandb = log_to_wandb
        self.num_epoch = num_epoch
        self.use_action_quantizer = use_action_quantizer
        
        ranges = np.load(os.path.join(data_path, "ranges.npy"), allow_pickle=True).item()
        action_space_low = ranges['command_space_low']
        action_space_high = ranges['command_space_high']
        print("action_space_low: ", action_space_low)
        print("action_space_high: ", action_space_high)
        self.action_dim = len(action_space_low)

        self.model = RT1(input_tensor_space = gym.spaces.Dict(
                {
                    'image': gym.spaces.Box(low=0.0, high=1.0, 
                                    shape=(1, image_height, image_width), dtype=np.float32),
                    'natural_language_embedding': gym.spaces.Box(low=-np.inf, high=np.inf, 
                                    shape=[embedding_dim], dtype=np.float32)
                }
            ), # observation space like dict. keys are image, natural_language_embedding
            output_tensor_space = gym.spaces.Dict(
                {   
                    'terminate_episode': gym.spaces.Discrete(2),
                    'commands': gym.spaces.Box(low=action_space_low, high=action_space_high, dtype=np.float32),
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
            use_action_quantizer = use_action_quantizer,
            condition_dim = 31,
            return_attention_scores = False,
            device=self.device,
        )

        # the pretrained Universal Sentence Encoder Large 5 is loaded from onnx file.
        self.sentence_encoder = load_onnx_model(use_path, use_cuda=device.type == "cuda")
        print("Load pretrained weights of Universal Sentence Encoder Large 5 from " + use_path)
        
        if use_action_quantizer:
            if action_quantizer_path is not None:
                self.model._action_tokenizer.load_dict(torch.load(action_quantizer_path))
                print("Load action quantizer from " + action_quantizer_path)
            # freeze the parameters of action quantizer.
            for name, param in self.model._action_tokenizer.named_parameters():
                param.requires_grad = False
            self.model._action_tokenizer.eval()
        
        # analysis of the model parameters.
        # model_analysis(self.model)
        # create dataset
        task_list = list(INSTRUCTION_DICT.keys())
        sub_datasets = []
        start_time = datetime.now()
        for task in task_list:
            task_path = os.path.join(data_path, task)
            if not os.path.exists(task_path):
                continue
            task_sub_datasets = [RT1Dataset(dataset_type='train', 
                                            dataset_path=os.path.join(task_path, episode_num), 
                                            time_sequence_length=self.time_sequence_length, 
                                            transform=transform_train) for episode_num in os.listdir(task_path) 
                                if os.path.exists(os.path.join(task_path, episode_num) + '/datalist.json')]
            # if len(sub_dataset) >= self.time_sequence_length:
            sub_datasets.extend(task_sub_datasets)
        end_time = datetime.now()
        # print the time used to load the dataset in seconds
        self.dataset = torch.utils.data.ConcatDataset(sub_datasets)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count(), drop_last=True)
        traj_num = len(sub_datasets)
        data_num = len(self.dataset)
        iter_per_epoch = data_num // batch_size
        print("Time used to load the dataset (seconds): ", (end_time - start_time).seconds)
        print("Total number of used trajectories: ", traj_num)
        print("Total number of used data: ", data_num)
        print("Number of iterations per epoch: ", iter_per_epoch)
        
        self.base_lr = 1e-4
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.base_lr)
        
        # cosine annealing learning rate scheduler with warm up.
        warm_up_iter = iter_per_epoch
        T_max = iter_per_epoch * num_epoch
        lr_max = self.base_lr
        lr_min = 1e-8

        lr_lambda = lambda cur_iter: cur_iter / warm_up_iter if cur_iter < warm_up_iter else \
                (lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos((cur_iter - warm_up_iter) / (T_max - warm_up_iter) * np.pi))) / self.base_lr

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=[lr_lambda])
        
        
        # daytime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        daytime = datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d-%H-%M-%S")
        exp_prefix = f'experiment-{daytime}'
        train_config_dict = {
                    'batch_size': batch_size,
                    'num_epoch': num_epoch,
                    'use_path': use_path,
                    'data_path': data_path,
                    'device': device,
                    'num_epoch': num_epoch,
                    'traj_num': traj_num,
                    'data_num': data_num,
                    'iter_per_epoch': iter_per_epoch,
                    'base_lr': self.base_lr,
                    'warm_up_iter': warm_up_iter,
                    'T_max': T_max,
                    'lr_max': lr_max,
                    'lr_min': lr_min,
                    'lr_lambda': lr_lambda
        }
        model_config_dict = {
                    'input_tensor_space': {
                        'image': {
                            'shape': (1, image_height, image_width),
                            'dtype': np.float32
                        },
                        'natural_language_embedding': {
                            'shape': [embedding_dim],
                            'dtype': np.float32
                        }
                    },
                    'output_tensor_space': {
                        'commands': {
                            'shape': (11,),
                            'dtype': np.float32
                        },
                        'terminate_episode': {
                            'shape': (),
                            'dtype': np.int64
                        }
                    },
                    'vocab_size': 256,
                    'num_layers': 8,
                    'layer_size': 256,
                    'num_heads': 6,
                    'feed_forward_size': 256,
                    'dropout_rate': 0.1,
                    'time_sequence_length': time_sequence_length,
                    'use_token_learner': True,
                    'return_attention_scores': False,
                    'image_height': image_height,
                    'image_width': image_width,
                    'embedding_dim': embedding_dim
        }
        if log_to_wandb:
            wandb.init(
                name=exp_prefix,
                project='quadruped_rt1',
                entity='milab_legged_vla',
                config={
                    "train_parameters": train_config_dict,
                    "model_parameters": model_config_dict
                }
                # group=group_name,
            )
        self.output_path = os.path.join(ROOT_PATH, 'outputs', daytime)
            
        self.timestep = 0
            
    def inference_one_time_step(self, images, instructions, network_state=None):
        self.model.eval()
        batch_size = images.shape[0]
        assert len(instructions) == batch_size
        sentence_embeddings = self.sentence_encoder.run(output_names=["outputs"], input_feed={"inputs": instructions})[0]
        sentence_embeddings = torch.from_numpy(sentence_embeddings).to(self.device)
        
        observation = {
            'image': images,
            'natural_language_embedding': sentence_embeddings
        }

        if network_state is None:
            network_state = {
                            'context_image_tokens': torch.randint(0, 256, (batch_size, self.time_sequence_length, 8, self.embedding_dim), device=self.device),
                            'action_tokens': torch.randint(0, 256, (batch_size, self.time_sequence_length, self.action_dim + 1), device=self.device),
                            # Stores where in the window we are.
                            # This value is within range [0, time_sequence_length + 1].
                            # When seq_idx == time_sequence_length, context_image_tokens and
                            # action_tokens need to be shifted to the left.
                            'seq_idx': torch.zeros((1, 1), dtype=torch.int64, device=self.device)
                            # Our data is like context_image_tokens + action_tokens + context_image_tokens + action_tokens + context_image_tokens ...
                            # 1 time step means [context_image_tokens + action_tokens]
                            # seq_idx means which time steps we are. But it is adjusted to time_sequence_length when it exceeds time_sequence_length.
                            }
        
        output_actions, network_state = self.model(observation, network_state)
        
        return output_actions, network_state
    
    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        print(f"Load pretrained weights of RT-1 from {model_path}")
    
    def save(self, index):
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        torch.save(self.model.state_dict(), f"{self.output_path}/model_epoch_{index}.pth")
        print(f"Save model to {self.output_path}/model_epoch_{index}.pth")
        
    def train(self):
        print("Start training...")
        for i in range(self.num_epoch):
            loss = self.train_epoch()
            print("Epoch: {}, Loss: {}".format(i + 1, loss))
            self.save(i + 1)
    
    def train_epoch(self):
        loss = 0
        batch_per_epoch = len(self.dataloader)
        for _, (images, instructions, commands, terminates, proprioceptions) in enumerate(self.dataloader):
            images = images.to(self.device)
            instructions = list(instructions)
            actions = {
                'commands': commands.to(self.device),
                'terminate_episode': terminates.to(self.device)
            }
            proprioceptions = proprioceptions.to(self.device)
            
            loss += self.train_batch(images, instructions, actions, proprioceptions)
        loss /= batch_per_epoch
        return loss
    
    def train_batch(self, images, instructions, actions, proprioceptions):
        self.model.train()
        batch_size = images.shape[0]
        assert len(instructions) == batch_size
        sentence_embeddings = self.sentence_encoder.run(output_names=["outputs"], input_feed={"inputs": instructions})[0]
        sentence_embeddings = torch.from_numpy(sentence_embeddings).unsqueeze(1).repeat(1, self.time_sequence_length, 1).to(self.device)

        observation = {
            'image': images,
            'natural_language_embedding': sentence_embeddings, 
            'proprioception': proprioceptions,
            'action': actions
        }
        network_state = {
                        'context_image_tokens': torch.randn(batch_size, self.time_sequence_length, 8, self.embedding_dim, device=self.device),
                        'action_tokens': torch.randn(batch_size, self.time_sequence_length, self.action_dim + 1, device=self.device),
                        # Stores where in the window we are.
                        # This value is within range [0, time_sequence_length + 1].
                        # When seq_idx == time_sequence_length, context_image_tokens and
                        # action_tokens need to be shifted to the left.
                        'seq_idx': torch.randint(0, self.time_sequence_length + 1, (batch_size, 1), dtype=torch.int64, device=self.device)
                        # Our data is like context_image_tokens + action_tokens + context_image_tokens + action_tokens + context_image_tokens ...
                        # 1 time step means [context_image_tokens + action_tokens]
                        # seq_idx means which time steps we are. But it is adjusted to time_sequence_length when it exceeds time_sequence_length.
                        }
        
        _, _, aux_info = self.model(observation, network_state)
        loss = aux_info['action_loss'].mean()
        # calculate the average loss of each action.
        action_loss = aux_info['action_loss'].mean(dim=[0, 1])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        # print(aux_info['action_labels'])
        # print(aux_info['action_predictions'])
        # print(aux_info['action_loss_mask'])
        # print(aux_info['action_loss'])
        
        if self.log_to_wandb:
            wandb.log({
                "total loss": loss.item(),
                "dx loss": action_loss[0].item(),
                "dy loss": action_loss[1].item(),
                "dyaw loss": action_loss[2].item(),
                "body height loss": action_loss[3].item(),
                "step frequency loss": action_loss[4].item(),
                "gait loss": action_loss[5:8].mean().item(),
                "footswing height loss": action_loss[8].item(),
                "pitch loss": action_loss[9].item(),
                "stance width loss": action_loss[10].item(),
                "terminate loss": action_loss[11].item(),
                "learning rate": self.optimizer.param_groups[0]['lr']
            })
        
        return loss.item()