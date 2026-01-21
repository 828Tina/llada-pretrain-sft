export PYTHONPATH="../:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=5 accelerate launch \
        --config_file configs/ddp.yaml --num_processes 1 \
        examples/qwen/pt.py \
        --configs configs/qwen2.5-100M-pt.yaml