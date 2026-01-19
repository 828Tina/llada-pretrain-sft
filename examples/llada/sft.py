"""
Local users
------------
- 1 GPU (4bit quant & LoRA, useful for testing):
    CUDA_VISIBLE_DEVICES=4 accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/llada/sft.py \
        --configs scripts/llada-8b-sft.yaml

CUDA_VISIBLE_DEVICES=3,5,6,7 accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml --num_processes 4 \
    examples/llada/sft.py \
    --lora True

CUDA_VISIBLE_DEVICES=3,5,6,7 accelerate launch \
    --config_file scripts/accelerate_configs/zero3.yaml --num_processes 4 \
    examples/llada/sft.py \
    --num_train_epochs 4 \
    --lora True

- 8 GPUs (FSDP):
    accelerate launch \
        --config_file scripts/accelerate_configs/fsdp.yaml \
        examples/llada/sft.py

Slurm users
# Note: run `mkdir logs` before running sbatch; and adjust
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 1 Node, 8 GPUs (FSDP):
    sbatch --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "fsdp" \
        --script_path "examples/llada/sft.py"

- 2 Nodes, 16 GPUs (FSDP):
    sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "fsdp" \
        --script_path "examples/llada/sft.py"
"""

import os
from dataclasses import dataclass, field
from functools import partial

import accelerate
import transformers

import dllm

import argparse
import yaml

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "/data/lxy/diffusion/llada-8b"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "/data/lxy/diffusion/data/alpaca-zh-gpt[train:2000,test:200]"
    load_preprocessed_data: bool = True # huggingface: load_from_disk 
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask the loss on the prompt tokens"},
    )


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = "/data/lxy/diffusion/output/llada-gpu1-epoch-3"
    group_by_length: bool = True
    run_name: str = "llada-alpaca-zh-epoch-3"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    eval_strategy: str = "steps"
    eval_steps: float = 100
    save_steps: float = 100


def train():
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    tmp_parser = argparse.ArgumentParser()
    tmp_parser.add_argument("--configs", required=True)
    tmp_args, _ = tmp_parser.parse_known_args()

    with open(tmp_args.configs, "r", encoding="utf-8") as f:
        yaml_dict = yaml.safe_load(f)

    model_args, data_args, training_args = parser.parse_dict(yaml_dict)

    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # ----- Model ------------------------------------------------------------------
    model = dllm.utils.get_model(model_args=model_args)
    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # ----- Dataset ----------------------------------------------------------------
    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_sft_dataset(
            data_args.dataset_args,
            load_preprocessed_data=data_args.load_preprocessed_data,
        )
        if not data_args.load_preprocessed_data:
            map_fn = partial(
                dllm.utils.default_mdlm_sft_map_fn,
                tokenizer=tokenizer,
                mask_prompt_loss=data_args.mask_prompt_loss,
            )
            dataset = dataset.map(
                map_fn,
                num_proc=data_args.num_proc,
                desc="Mapping dataset to SFT format",
            )
        # truncate / filter long sequences if needed
        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    # ----- Training --------------------------------------------------------------
    accelerate.PartialState().wait_for_everyone()
    logger.info("Start training...")
    trainer = dllm.core.trainers.MDLMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=(
            dllm.utils.NoAttentionMaskWrapper(  # padded <eos_token> should be visible
                transformers.DataCollatorForSeq2Seq(
                    tokenizer,
                    return_tensors="pt",
                    padding=True,
                    label_pad_token_id=tokenizer.pad_token_id,  # finetune on padded <eos_token>
                ),
            )
        ),
    )
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
