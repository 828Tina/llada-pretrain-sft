#!/usr/bin/env bash
# ===== Mandatory for proper import and evaluation =====
export PYTHONPATH=../:$PYTHONPATH             
export HF_ALLOW_CODE_EVAL=1                 # Allow code evaluation
export HF_DATASETS_TRUST_REMOTE_CODE=True   # For cmmlu dataset

# ===== Optional but recommended for stability and debugging =====
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1    # Enable async error handling for multi-GPU communication to avoid deadlocks
export NCCL_DEBUG=warn                      # Show NCCL warnings for better diagnosis without flooding logs
export TORCH_DISTRIBUTED_DEBUG=DETAIL       # Provide detailed logging for PyTorch distributed debugging
# =======================

# CUDA_VISIBLE_DEVICES=4 accelerate launch --num_processes 1 \
#     dllm/pipelines/llada/eval.py \
#     --tasks ceval-valid \
#     --model llada \
#     --apply_chat_template \
#     --num_fewshot 5 \
#     --output_path /data/lxy/diffusion/eval/llada-ceval \
#     --log_samples \
#     --max_batch_size 4 \
#     --model_args "pretrained=/data/lxy/diffusion/llada-8b,is_check_greedy=False,mc_num=1,max_length=1024,steps=256,block_size=64,cfg=0.0"


CUDA_VISIBLE_DEVICES=5 accelerate launch --num_processes 1 \
    dllm/pipelines/llada/eval.py \
    --tasks cmmlu \
    --model llada \
    --apply_chat_template \
    --output_path /data/lxy/diffusion/eval/llada/llada-cmmlu/llada-8b-epoch-3/test \
    --log_samples \
    --max_batch_size 4 \
    --model_args "pretrained=/data/lxy/diffusion/output/merge-llada-8b-alpaca-zh-gpt-epoch-3,is_check_greedy=False,mc_num=1,max_length=1024,steps=256,block_size=64,cfg=0.0"