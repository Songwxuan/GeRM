import torch
import argparse
import wandb
import numpy as np
import os
import pytz
import gym
from datetime import datetime
from pytorch_robotics_transformer.tokenizers.action_quantizer import ActionQuantizer
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from multiprocessing import cpu_count
from __init__ import ROOT_PATH
from dataset import StateCommandDataset

def experiment(variant):
    print(ROOT_PATH)
    output_path = os.path.join(ROOT_PATH, 'outputs/action_quantizer')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    log_to_wandb = variant.get('log_to_wandb', True)
    batch_size = variant.get('batch_size', 512)
    num_epoch = variant.get('epochs', 100)
    num_embeddings = variant.get('num_embeddings', 256)
    embedding_dim = variant.get('embedding_dim', 16)
    encoder_hidden_dims = variant.get('encoder_hidden_dims', [512, 256])
    activation = variant.get('activation', 'ELU')
    lr = variant.get('learning_rate', 3e-4)

    data_path = os.path.join(ROOT_PATH, "dataset_info")

    ranges = np.load(os.path.join(data_path, "ranges.npy"), allow_pickle=True).item()
    action_space_low = ranges['command_space_low']
    action_space_high = ranges['command_space_high']
    proprioception_mean = ranges['proprio_mean']
    proprioception_std = ranges['proprio_std']
    print("action_space_low: ", action_space_low)
    print("action_space_high: ", action_space_high)
    print("proprioception_mean: ", proprioception_mean)
    print("proprioception_std: ", proprioception_std)
    action_dim = len(action_space_low)
    condition_dim = len(proprioception_mean)
    
    action_quantizer = ActionQuantizer(action_space=gym.spaces.Dict(
                {   
                    'terminate_episode': gym.spaces.Discrete(2),
                    'commands': gym.spaces.Box(low=action_space_low, high=action_space_high, dtype=np.float32),
                }
            ),
                 vocab_size=num_embeddings,
                 condition_dim=condition_dim,
                 embedding_dim=embedding_dim,
                 encoder_hidden_dims=encoder_hidden_dims,
                 activation=activation
    ).to(device)

    # create dataset
    task_list = list(INSTRUCTION_DICT.keys())
    sub_datasets = []
    start_time = datetime.now()
    for task in task_list:
        task_path = os.path.join(data_path, task)
        if not os.path.exists(task_path):
            continue
        for episode_num in os.listdir(task_path):
            episode_path = os.path.join(task_path, episode_num)
            if not os.path.exists(episode_path + '/datalist.json'):
                continue
            sub_dataset = StateCommandDataset(dataset_type='train', dataset_path=episode_path)
            # if len(sub_dataset) >= self.time_sequence_length:
            sub_datasets.append(sub_dataset)
    end_time = datetime.now()
    # print the time used to load the dataset in seconds
    dataset = torch.utils.data.ConcatDataset(sub_datasets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count())
    traj_num = len(sub_datasets)
    data_num = len(dataset)
    iter_per_epoch = data_num // batch_size
    print("Time used to load the dataset (seconds): ", (end_time - start_time).seconds)
    print("Total number of used trajectories: ", traj_num)
    print("Total number of used data: ", data_num)
    print("Number of iterations per epoch: ", iter_per_epoch)

    optimizer = torch.optim.AdamW(action_quantizer.parameters(), lr=lr)

    daytime = datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d-%H-%M-%S")
    exp_prefix = f'experiment-{daytime}'
    train_config_dict = {
        "batch_size": batch_size,
        "num_epoch": num_epoch,
        "traj_num": traj_num,
        "data_num": data_num,
        "iter_per_epoch": iter_per_epoch,
        "learning_rate": lr
    }
    model_config_dict = {
        "action_dim": action_dim,
        "condition_dim": condition_dim,
        "action_space_low": action_space_low,
        "action_space_high": action_space_high,
        "proprioception_mean": proprioception_mean,
        "proprioception_std": proprioception_std,
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "encoder_hidden_dims": encoder_hidden_dims,
        "activation": activation
    }
    if log_to_wandb:
        wandb.init(
                name=exp_prefix,
                project='action_quantizer',
                entity='milab_legged_vla',
                config={
                    "train_parameters": train_config_dict,
                    "model_parameters": model_config_dict
                }
                # group=group_name,
            )
        
    print("Start training...")
    for epoch in range(num_epoch):
        epoch_loss = 0
        for _, (actions, conditions) in enumerate(dataloader):
            actions = actions.to(device)
            conditions = conditions.to(device)
            input = {
                'actions': actions,
                'conditions': conditions
            }
            output = action_quantizer(input)
            
            reconstruction_loss = output['reconstruction_loss']
            q_latent_loss = output['q_latent_loss']
            e_latent_loss = output['e_latent_loss']
            contra_loss = output['contrastive_loss']
            perplexity = output['perplexity']
            loss = reconstruction_loss + q_latent_loss + e_latent_loss + contra_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() / iter_per_epoch

            if log_to_wandb:
                wandb.log({
                    "total loss": loss.item(),
                    "reconstruction loss": reconstruction_loss.item(),
                    "q latent loss": q_latent_loss.item(),
                    "e latent loss": e_latent_loss.item(),
                    "contrastive loss": contra_loss.item(),
                    "perplexity": perplexity.item()
                })
        print("Epoch: {}, Loss: {}".format(epoch, epoch_loss))
        if epoch % 100 == 0:
            torch.save(action_quantizer.state_dict(), os.path.join(output_path, "action_quantizer_{}.pth".format(epoch)))
    torch.save(action_quantizer.state_dict(), os.path.join(output_path, "action_quantizer.pth"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--sequence_length', type=int, default=6)
    parser.add_argument('--log_to_wandb', type=bool, default=False)
    
    args = parser.parse_args()
    
    experiment(variant=vars(args))