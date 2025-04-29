import torch
import torch.distributed as dist
import argparse
import wandb
import os
# from agent import Agent
from agent_ddp import AgentDDP
from __init__ import ROOT_PATH

def experiment(variant, rank, local_rank, world_size, device):
    output_path = os.path.join(ROOT_PATH, 'outputs')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    log_to_wandb = variant.get('log_to_wandb', False)
    batch_size = variant.get('batch_size', 4)
    num_epoch = variant.get('epochs', 10)
    time_sequence_length = variant.get('sequence_length', 6)
    use_action_quantizer = variant.get('use_action_quantizer', False)
    
    image_height = 300
    image_width = 300
    embedding_dim = 512
    # action_dim = 11
    use_path = os.path.join(ROOT_PATH, "pytorch_robotics_transformer/universal_sentence_encoder_large_5_onnx/model.onnx")
    data_path = os.path.join(ROOT_PATH, "dataset_info")
    action_quantizer_path = None

    sim_data_ratio = variant.get('sim_data_ratio', 1.0)
    
    agent = AgentDDP(
                rank,
                local_rank,
                world_size,
                batch_size,
                time_sequence_length,
                image_height,
                image_width,
                embedding_dim,
                use_path,
                data_path,
                action_quantizer_path,
                device,
                log_to_wandb,
                use_action_quantizer,
                num_epoch,
                sim_data_ratio)
    

    agent.train()
    

def setup_DDP(backend="nccl", verbose=False):
    """
    We don't set ADDR and PORT in here, like:
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '12355'
    Because program's ADDR and PORT can be given automatically at startup.
    E.g. You can set ADDR and PORT by using:
        python -m torch.distributed.launch --master_addr="192.168.1.201" --master_port=23456 ...

    You don't set rank and world_size in dist.init_process_group() explicitly.

    :param backend:
    :param verbose:
    :return:
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # If the OS is Windows or macOS, use gloo instead of nccl
    dist.init_process_group(backend=backend)
    # set distributed device
    device = torch.device("cuda:{}".format(local_rank))
    if verbose:
        print(f"local rank: {local_rank}, global rank: {rank}, world size: {world_size}")
    return rank, local_rank, world_size, device


if __name__ == "__main__":
    os.environ["WANDB_API_KEY"] = "3cc75ad549e94669d4b230f5cdea68473279dc08"
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--sequence_length', type=int, default=6)
    parser.add_argument('--log_to_wandb', type=bool, default=False)
    parser.add_argument('--sim_data_ratio', type=float, default=1.0)
    parser.add_argument('--use_action_quantizer', type=bool, default=False)
    parser.add_argument('--local-rank', type=int, default=0)
    
    args = parser.parse_args()
    
    rank, local_rank, world_size, device = setup_DDP(verbose=True)
    if local_rank == 0:
        print("Start DDP...")
    
    experiment(variant=vars(args), rank=rank, local_rank=local_rank, world_size=world_size, device=device)