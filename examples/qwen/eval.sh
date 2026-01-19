# -------- Evalscope Evaluation Script --------
# This script is used to evaluate language models using the Evalscope framework.

# Usage:
#     bash eval.sh
# ---------------------------------------------
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_SKIP_VERIFY=1
export TRANSFORMERS_OFFLINE=1

#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=7 evalscope eval \
 --model /data/lxy/diffusion/output/merge-qwen2.5-7b-alpaca-zh-gpt-epoch-200 \
 --generation-config '{"max_length":1024}' \
 --datasets ceval \
 --dataset-args '{"ceval":{"local_path":"/data/lxy/diffusion/eval/data/ceval"}}' \
 --work-dir /data/lxy/diffusion/eval/qwen-ceval


export VLLM_USE_MODELSCOPE=True 
CUDA_VISIBLE_DEVICES=6 python -m vllm.entrypoints.openai.api_server \
 --model /data/lxy/diffusion/output/merge-qwen2.5-7b-alpaca-zh-gpt-epoch-200 \
 --served-model-name qwen2.5 \
 --port 8000

evalscope eval \
 --model qwen2.5 \
 --eval-type service \
 --eval-batch-size 4 \
 --api-url http://127.0.0.1:8000/v1 \
 --api-key EMPTY \
 --generation-config '{"max_length":1024}' \
 --datasets ceval \
 --dataset-args '{"ceval":{"local_path":"/data/lxy/diffusion/eval/data/ceval"}}' \
 --work-dir /data/lxy/diffusion/eval/qwen-ceval