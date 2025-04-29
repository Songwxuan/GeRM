source /opt/conda/bin/activate base
cd /dingpengxiang/Wenxuan/germ

python -u -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=1234 \
    experiment.py \
    --batch_size=16 \
    --epochs=10 \
    --sequence_length=6 \
    --sim_data_ratio=0.05 \
    --log_to_wandb=False  # sim_data_ratio = 1, 0.125, 0.015625 batch = 32 epoch = 20 nproc_per_node=4