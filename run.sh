export MODEL_PATH='/data/chengshuang/NAACL/lost-in-the-middle/LLM_Model/Llama-2-7b-hf'
export SAVE_PATH='output'
export MASTER_ADDR="localhost"
export MASTER_PORT="1231"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export WANDB_DISABLED=true
wandb offline

python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=2 --use_env train_math.py \
    --model_name_or_path $MODEL_PATH \
    --data_path "/data/chengshuang/CL/MetaMath/data/train/MetaMathQA/MetaMathQA-395K.json" \
    --data_length 10000000 \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --method "ASCII" \
    --threshold: 0.5 \
    --no_cluster: True \
    --n_cluster: 32
# --fsdp "no_shard" \
# --fsdp_offload_params True \
# --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
# --fsdp "no_shard" full_shard auto_wrap
# python eval_gsm8k.py --model $SAVE_PATH --data_path ./data/test/GSM8K_test.jsonl
# python eval_math.py --model $SAVE_PATH --data_path ./data/test/MATH_test.jsonl
