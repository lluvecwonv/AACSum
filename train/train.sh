master_port=18767
lr=2e-5
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$master_port train.py --config-name=finetune_t5.yaml split=${split} batch_size=2 gradient_accumulation_steps=1 lr=${lr}


master_port=18762
lr=2e-5
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$master_port train.py --config-name=finetune_roberta.yaml split=${split} batch_size=4 gradient_accumulation_steps=4 lr=${lr}