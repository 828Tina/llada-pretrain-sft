#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    CUDA_VISIBLE_DEVICES=5 python chat.py \
        --model_name_or_path /data/lxy/diffusion/output/merge-qwen2.5-7b-alpaca-zh-gpt-epoch-20
"""
import argparse
import sys
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen 交互式对话")
    parser.add_argument("--model_name_or_path", required=True,
                        help="ModelScope 模型 ID 或本地路径")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="生成的最大新tokens数")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="采样温度")
    return parser.parse_args()

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map="auto"
    )

    history = [{"role": "system",
                "content": "You are a helpful assistant."}]
    try:
        while (user := input("\n>>> ")).strip() != "exit":
            history.append({"role": "user", "content": user})
            prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=args.temperature)
            reply = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            print(reply)
            history.append({"role": "assistant", "content": reply})
    except (KeyboardInterrupt, EOFError):
        print("\nBye~")

if __name__ == "__main__":
    main(parse_args())