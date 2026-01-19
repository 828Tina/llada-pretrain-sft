CUDA_VISIBLE_DEVICES=4 accelerate launch \
        --config_file configs/ddp.yaml \
        --num_processes 1 \
        examples/qwen/sft.py \
        --configs configs/qwen2.5-7b-alpaca.yaml
