export PYTHONPATH="../:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=5 accelerate launch \
        --config_file configs/ddp.yaml --num_processes 1 \
        examples/llada/pt.py \
        --configs configs/llada-100M-pt.yaml