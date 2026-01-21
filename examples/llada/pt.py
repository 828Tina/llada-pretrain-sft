"""
Local users
------------
- 1 GPU (4bit quant & LoRA, useful for testing):
    CUDA_VISIBLE_DEVICES=5 accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        pt.py \
        --config_file scripts/parameters_configs/llada-100M-pt.yaml

- 8 GPUs (FSDP):
    accelerate launch \
        --config_file scripts/accelerate_configs/fsdp.yaml \
        examples/llada/pt.py

Slurm users
# Note: run `mkdir logs` before running sbatch; and adjust
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 24 Nodes, 192 GPUs (FSDP):
    sbatch --nodes=24 --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "fsdp" \
        --script_path "examples/llada/pt.py"
"""。。。


import functools
import os
from dataclasses import dataclass, field

import accelerate
import torch
import transformers

import dllm

import argparse
import yaml

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    # Uses only the configuration from model_name_or_path to initialize the model from scratch
    model_name_or_path: str = (
        "/data/lxy/diffusion/llada-100M"  # "inclusionAI/LLaDA-MoE-7B-A1B-Base"
    )


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "/data/lxy/diffusion/data/c4-en-shuffled[train:1000_000,test:1000]"
    text_field: str = "text"
    streaming: bool = False
    num_proc: int = 8
    drop_tail: bool = True
    max_length: int = 1024
    load_preprocessed_data: bool = True 
    insert_eos: bool = field(
        default=True,
        metadata={
            "help": "False when adjacent samples from the datasets are semantically coherent."
        },
    )
    random_length_ratio: float = field(
        default=0.01,
        metadata={
            "help": (
                "The probability of randomly cut sequences during training. "
                "See https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md#pre-training for reference."
            )
        },
    )


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = (
        "/data/lxy/diffusion/output/llada-pt-c4-500Mtokens-epoch-1"
    )
    run_name: str = "llada-pt-c4-500Mtokens-epoch-1"
    learning_rate: float = 3e-4
    warmup_steps: int = 2000
    # max_steps: int = 500
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    logging_steps: float = 20
    eval_strategy: str = "steps"
    eval_steps: float = 200
    save_steps: float = 1000


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
    # necessary for streaming dataset
    if data_args.streaming:
        training_args.accelerator_config.dispatch_batches = False
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # ----- Model ------------------------------------------------------------------
    # initialize model weights from scratch
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    with dllm.utils.init_device_context_manager():
        model = transformers.AutoModel.from_config(
            config, dtype=torch.bfloat16, init_params=True
        )

    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    # ----- Optional PEFT: LoRA ----------------------------------------------------
    model = dllm.utils.load_peft(model=model, model_args=model_args)

    # ----- Dataset ----------------------------------------------------------------
    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_pt_dataset(
            data_args.dataset_args,
            streaming=data_args.streaming,
            load_preprocessed_data=data_args.load_preprocessed_data,
        )
        dataset = dataset.map(
            functools.partial(
                dllm.utils.tokenize_and_group,
                tokenizer=tokenizer,
                text_field=data_args.text_field,
                seq_length=data_args.max_length,
                insert_eos=data_args.insert_eos,
                drop_tail=data_args.drop_tail,
            ),
            batched=True,
            remove_columns=dataset["train"].column_names,
            **({} if data_args.streaming else {"num_proc": data_args.num_proc}),
            **({} if data_args.streaming else {"desc": "Mapping dataset to PT format"}),
        )
        if data_args.streaming:
            dataset = dataset.shuffle(seed=training_args.seed)

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
            dllm.utils.RandomTruncateWrapper(
                transformers.DataCollatorForSeq2Seq(
                    tokenizer,
                    return_tensors="pt",
                    padding=True,
                ),
                random_length_ratio=data_args.random_length_ratio,
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
