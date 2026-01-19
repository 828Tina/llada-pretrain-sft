export PYTHONPATH="../:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=4 accelerate launch \
        --config_file configs/ddp.yaml --num_processes 1 \
        examples/llada/sft.py \
        --configs configs/llada-8b-sft.yaml