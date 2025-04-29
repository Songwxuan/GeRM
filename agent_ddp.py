import gym
import torch
import numpy as np
import pytz
import os
import wandb
import json
import torch.distributed as dist
import random
from datetime import datetime
from dataset import RT1Dataset
from dataset_fail import RT1Dataset_fail
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from model_analysis import model_analysis
from pytorch_robotics_transformer.transformer_network import TransformerNetwork as RT1
from pytorch_robotics_transformer.universal_sentence_encoder_large_5_onnx import load_onnx_model
from multiprocessing import cpu_count
from __init__ import ROOT_PATH, DAY_TIME

def print_only_rank0(*args):
    if dist.get_rank() == 0:
        print(*args)

class AgentDDP:
    def __init__(self,
                 rank,
                 local_rank,
                 world_size,
                 batch_size: int=2,
                 time_sequence_length: int=6,
                 image_height: int=256,
                 image_width: int=320,
                 embedding_dim: int=512,
                 use_path: str=os.path.join(ROOT_PATH, 'pytorch_robotics_transformer/universal_sentence_encoder_large_5_onnx/model.onnx'),
                #  data_path: str=os.path.join(ROOT_PATH, 'dataset_info'),
                 data_path = '/dingpengxiang/Wenxuan/walk-these-ways/datasets/auto',
                 action_quantizer_path: str=os.path.join(ROOT_PATH, 'outputs/action_quantizer/action_quantizer.pth'),
                 device=torch.device("cpu"),
                 log_to_wandb: bool=False,
                 use_action_quantizer: bool=False,
                 num_epoch: int=10,
                 sim_data_ratio: float=0.05,
                 gamma = 0.99,
                 output_path: str=os.path.join(ROOT_PATH, 'your_output_path', DAY_TIME)
                 ):
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.time_sequence_length = time_sequence_length
        self.image_height = image_height
        self.image_width = image_width
        self.embedding_dim = embedding_dim
        self.device = device
        self.log_to_wandb = log_to_wandb
        self.num_epoch = num_epoch
        self.use_action_quantizer = use_action_quantizer
        self.output_path = output_path

        ranges = np.load(os.path.join(data_path, "ranges.npy"), allow_pickle=True).item()
        action_space_low = ranges['command_space_low']
        action_space_high = ranges['command_space_high']
        print_only_rank0("action_space_low: ", action_space_low)
        print_only_rank0("action_space_high: ", action_space_high)
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
            num_heads = 8,
            feed_forward_size = 256, # This corresponds to d_model which is embedding dimension of each token in transformer part.
            dropout_rate = 0.1,
            time_sequence_length = time_sequence_length + 1,
            # action_order: Optional[List[str]] = None,
            action_order = ['terminate_episode', 'commands'],
            use_token_learner = True,
            use_action_quantizer = use_action_quantizer,
            condition_dim = 31,
            return_attention_scores = False,
            device=self.device,

        )
        
        if use_action_quantizer:
            if action_quantizer_path is not None:
                self.model._action_tokenizer.load_dict(torch.load(action_quantizer_path))
                print_only_rank0("Load action quantizer from " + action_quantizer_path)
            # freeze the parameters of action quantizer.
            for name, param in self.model._action_tokenizer.named_parameters():
                param.requires_grad = False
            self.model._action_tokenizer.eval()
        
        self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        # the pretrained Universal Sentence Encoder Large 5 is loaded from onnx file.
        self.sentence_encoder = load_onnx_model(use_path, use_cuda=True, device_id=local_rank)
        print_only_rank0("Load pretrained weights of Universal Sentence Encoder Large 5 from " + use_path)
        
        # analysis of the model parameters.
        # model_analysis(self.model)
        # create dataset
        start_time = datetime.now()
        #wx:原
        # real_file = open(os.path.join(ROOT_PATH, 'dataset_info/quadruped_data_info.json'), 'r')
        # real_dict = json.load(real_file)
        # real_dataset = [RT1Dataset(traj_path=key, sample_dict=real_dict[key], dataset_type='real', time_sequence_length=time_sequence_length + 1) for key in real_dict.keys()]
        real_dataset = []
        

        #demonstration数据加载
        demon_file = open(os.path.join(ROOT_PATH, 'dataset_info/sim_quadruped_data_info.json'), 'r')
        demon_dict = json.load(demon_file)
        demon_dataset = [RT1Dataset(traj_path=key, sample_dict=demon_dict[key], dataset_type='sim', time_sequence_length=time_sequence_length + 1) for key in demon_dict.keys() if 'unload' not in key and 'crawl' not in key]

        # fail数据加载
        start_time_sim = datetime.now()
        fail_dict = {}
        # 指定的文件夹路径
        folder_path = "/dingpengxiang/Wenxuan/walk-these-ways/datasets/auto/sim_json_path/sim"
        #WX:自己写了个能接收多个json文件的代码
        # 使用 os.walk() 遍历文件夹中的所有文件
        for rootpath, dirs, files in os.walk(folder_path):
            for filename in files:
                # 检查文件是否以 .json 结尾
                if filename.endswith('.json'):
                    # 构建文件的绝对路径
                    abs_file_path = os.path.join(rootpath, filename)
                    # 打开文件并加载 JSON 数据
                    with open(abs_file_path, 'r') as file:
                        data = json.load(file)
                    # 将加载的数据合并到字典中
                    fail_dict.update(data)
        # fail_dataset = [RT1Dataset_fail(traj_path=key, sample_dict=fail_dict[key], dataset_type='sim', time_sequence_length=time_sequence_length + 1) for key in fail_dict.keys() if 'unload' not in key]
        fail_dataset = []
        for key in fail_dict.keys():
            if 'unload' not in key:
                sample_dict = fail_dict[key] #sub_key
                # 检查 is_success 是否为 True，如果是则跳过该项数据
                if sample_dict['0']['is_success'][0]:
                    continue
                dataset_fail = RT1Dataset_fail(traj_path=key, sample_dict=sample_dict, dataset_type='sim', time_sequence_length=time_sequence_length + 1)
                fail_dataset.append(dataset_fail)


        end_time_sim = datetime.now()
        print_only_rank0('fail dataset loaded, time: ', (end_time_sim - start_time_sim).seconds)

        random.seed(0)
        random.shuffle(fail_dataset)
        random.shuffle(demon_dataset)
        self.sim_data_ratio = sim_data_ratio
        demon_dataset = demon_dataset[:int(len(demon_dataset) * sim_data_ratio)]
        fail_dataset = fail_dataset[:int(len(fail_dataset) * 0.02)]
        end_time = datetime.now()
        # print the time used to load the dataset in seconds
        self.dataset = torch.utils.data.ConcatDataset(demon_dataset + fail_dataset)
        self.sampler = DistributedSampler(self.dataset, shuffle=True, drop_last=True)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, sampler=self.sampler, num_workers=cpu_count() // world_size)
        traj_num = len(fail_dataset) + len(demon_dataset)
        data_num = len(self.dataset)
        iter_per_epoch = data_num // (batch_size * world_size)
        print_only_rank0("Time used to load the dataset (seconds): ", (end_time - start_time).seconds)
        print_only_rank0("Total number of used trajectories: ", traj_num)
        print_only_rank0("Total number of demon trajectories: ", len(demon_dataset))
        print_only_rank0("Total number of fail trajectories: ", len(fail_dataset))

        print_only_rank0("Total number of used data: ", data_num)
        print_only_rank0("Number of iterations per epoch: ", iter_per_epoch)
        
        self.base_lr = 3e-4
        #import pdb; pdb.set_trace()

        for name, paras in self.model.named_parameters():
            if 'ema_model' in name:
                paras.requires_grad = False

        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.base_lr)
        #冻结其中的ema_model
        
        # # cosine annealing learning rate scheduler with warm up.
        warm_up_iter = iter_per_epoch
        T_max = iter_per_epoch * num_epoch
        lr_max = self.base_lr
        lr_min = 1e-8

        lr_lambda = lambda cur_iter: cur_iter / warm_up_iter if cur_iter < warm_up_iter else \
                (lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos((cur_iter - warm_up_iter) / (T_max - warm_up_iter) * np.pi))) / self.base_lr

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=[lr_lambda])
        # self.scheduler = torch.optim.lr_scheduler.ConstantLR(torch.optim.AdamW(self.model.parameters(), lr=self.base_lr))
        self.gamma = gamma
        
        # daytime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        daytime = datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d-%H-%M-%S")
        exp_prefix = f'experiment-{daytime}'
        train_config_dict = {
                    'batch_size': batch_size,
                    'num_epoch': num_epoch,
                    'use_path': use_path,
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
                    'sim_data_ratio': sim_data_ratio,
                    'discounted_rate': gamma,
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
                    #WX:6改成8
                    'num_heads': 8,
                    'feed_forward_size': 256,
                    'dropout_rate': 0.1,
                    'time_sequence_length': time_sequence_length,
                    'use_token_learner': True,
                    'return_attention_scores': False,
                    'image_height': image_height,
                    'image_width': image_width,
                    'embedding_dim': embedding_dim
        }
            
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
                            'seq_idx': torch.zeros((1, 1), dtype=torch.int64, device=self.device),
                            # Our data is like context_image_tokens + action_tokens + context_image_tokens + action_tokens + context_image_tokens ...
                            # 1 time step means [context_image_tokens + action_tokens]
                            # seq_idx means which time steps we are. But it is adjusted to time_sequence_length when it exceeds time_sequence_length.
                            'returns': returns,
                            'rewards': rewards,
                            'idx': idx
                            }
        
        output_actions, network_state = self.model(observation, network_state)
        
        return output_actions, network_state
    
    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        print_only_rank0(f"Load pretrained weights of RT-1 from {model_path}")
    
    def save(self, index):
        if self.local_rank == 0:
            if not os.path.exists(self.output_path):
                os.mkdir(self.output_path)
            torch.save(self.model.state_dict(), f"{self.output_path}/model_epoch_{index}.pth")
            print_only_rank0(f"Save model to {self.output_path}/model_epoch_{index}.pth")
        
    def train(self):
        print_only_rank0("Start training...")
        for i in range(self.num_epoch):
            self.dataloader.sampler.set_epoch(i)
            loss = self.train_epoch()
            print_only_rank0("Epoch: {}, Loss: {}".format(i + 1, loss))
            self.save(i + 1)
    
    def train_epoch(self):
        loss = 0
        batch_per_epoch = len(self.dataloader)
        for _, (images, instructions, commands, terminates, masks, idx, is_success) in enumerate(self.dataloader):
            images = images.to(self.device)
            idx = idx.to(self.device)
            is_success = is_success.to(self.device)
            terminates = terminates.to(self.device)
            instructions = list(instructions)
            actions = {
                'commands': commands.to(self.device),
                'terminate_episode': terminates.to(self.device)
            }
            # proprioceptions = proprioceptions.to(self.device)
            proprioceptions = None
            masks = masks.to(self.device)
            is_success = is_success.unsqueeze(-1) #(bs,1) (4, 1)
            returns = self.gamma ** idx * is_success # (bs,t) (4, 7)
            # terminate　(bs,t) (4, 7)
            rewards = terminates * is_success #(bs,t) (4, 7)
            # print("rewards的值为", rewards)
            rewards = rewards.unsqueeze(-1)#(bs,t,1)
            returns = returns.unsqueeze(-1).repeat(1, 1, 12)#(bs,t,action)(4, 7, 12)

            # import pdb;
            # pdb.set_trace()
            
            batch_loss = self.train_batch(images, instructions, actions, proprioceptions, masks, idx, returns, rewards)
            loss += batch_loss
        loss /= batch_per_epoch
        return loss
    
    def train_batch(self, images, instructions, actions, proprioceptions, masks, idx, returns, rewards):
        self.model.train()
        batch_size = images.shape[0]
        assert len(instructions) == batch_size
        sentence_embeddings = self.sentence_encoder.run(output_names=["outputs"], input_feed={"inputs": instructions})[0]
        sentence_embeddings = torch.from_numpy(sentence_embeddings).unsqueeze(1).repeat(1, self.time_sequence_length + 1, 1).to(self.device)

        observation = {
            'image': images,
            'natural_language_embedding': sentence_embeddings, 
            'proprioception': proprioceptions,
            'action': actions
        }
        network_state = {
                        'context_image_tokens': torch.randn(batch_size, self.time_sequence_length + 1, 8, self.embedding_dim, device=self.device),
                        'action_tokens': torch.randn(batch_size, self.time_sequence_length + 1, self.action_dim + 1, device=self.device),
                        # Stores where in the window we are.rewards
                        # This value is within range [0, time_sequence_length + 1].
                        # When seq_idx == time_sequence_length, context_image_tokens and
                        # action_tokens need to be shifted to the left.
                        'seq_idx': torch.randint(0, self.time_sequence_length + 2, (batch_size, 1), dtype=torch.int64, device=self.device),
                        # Our data is like context_image_tokens + action_tokens + context_image_tokens + action_tokens + context_image_tokens ...
                        # 1 time step means [context_image_tokens + action_tokens]
                        # seq_idx means which time steps we are. But it is adjusted to time_sequence_length when it exceeds time_sequence_length.
                        'returns': returns,
                        'rewards': rewards,
                        'idx': idx
                        }
        # print("rewards值为", rewards)
        # print("returns值为", returns)
        _, _, aux_info = self.model(observation, network_state)
        aux_info['action_loss'][:, :, 4:] *= masks[:, :-1]
        loss = aux_info['action_loss'].mean() + aux_info['action_regularization_loss'].mean() + aux_info['moe_loss'].mean()
        # calculate the average loss of each action.
        action_loss = aux_info['action_loss'].mean(dim=[0, 1])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        # print_only_rank0(aux_info['action_labels'])
        # print_only_rank0(aux_info['action_predictions'])
        # print_only_rank0(aux_info['action_loss_mask'])
        # print_only_rank0(aux_info['action_loss'])
        
        # Wandb part
        # if self.log_to_wandb and dist.get_rank() == 0:
        #     if action_loss.shape[0] == 12:
        #         wandb.log({
        #             "terminate loss": action_loss[0].item(),
        #             "total loss": loss.item(),
        #             "dx loss": action_loss[1].item(),
        #             "dy loss": action_loss[2].item(),
        #             "dyaw loss": action_loss[3].item(),
        #             "body height loss": action_loss[4].item(),
        #             "step frequency loss": action_loss[5].item(),
        #             "gait loss": action_loss[6:9].mean().item(),
        #             "footswing height loss": action_loss[9].item(),
        #             "pitch loss": action_loss[10].item(),
        #             "stance width loss": action_loss[11].item(),
        #             "learning rate": self.optimizer.param_groups[0]['lr']
        #         })
        #     elif action_loss.shape[0] == 2:
        #         wandb.log({
        #             "total loss": loss.item(),
        #             "command_loss": action_loss[0].item(),
        #             "terminate loss": action_loss[1].item(),
        #             "learning rate": self.optimizer.param_groups[0]['lr']
        #         })
        
        return loss.item()
