CUDA_VISIBLE_DEVICES=6 accelerate launch \
        --config_file configs/ddp.yaml \
        --num_processes 1 \
        sft.py \
        --config_file configs/qwen2.5-7b-alpaca.yaml
