import transformers
from dataclasses import dataclass, field
import torch
import utils
import os
import argparse,yaml
import accelerate
import functools
import logging

os.environ["SWANLAB_PROJECT"]="llada-sft"

# ----- Arguments -------------------------

@dataclass
class ModelArguments:
    model_name_or_path: str = None # overwrite this
    dtype: str = "bfloat16"
    bias: str = "none"
    modules_to_save: str = None


@dataclass
class DataArguments:
    dataset_args: str = None # overwrite this
    text_field: str = "text"
    streaming: bool = False
    num_proc: int = 8
    drop_tail: bool = True
    max_length: int = 1024
    insert_eos: bool = field(
        default=True,
        metadata={
            "help": "False when adjacent samples from the datasets are semantically coherent."
        },
    )
    disable_caching: bool = False
    num_proc: int = 8
    truncation: str = field(
        default="right",
        metadata={
            "help": (
                'The truncation strategy to use ("filter" or "right"). '
                '"filter" only keeps sequences that are shorter than max_length; '
                '"right" only keeps the rightmost max_length tokens for each sequence.'
            )
        },
    )
    

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = None  # overwrite this
    report_to: str = "swanlab"
    run_name: str = "test-1"
    overwrite_output_dir: bool = True
    seed: int = 42
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 2000
    bf16: bool = True
    num_train_epochs: float = 6
    logging_steps: float = 10
    eval_on_start: bool = False
    eval_strategy: str = "steps"
    eval_steps: float = 0.25
    save_steps: float = 0.25
    save_only_model: bool = True
    save_total_limit: int = 2


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
    # ----- print args and parameters --------------------
    print("模型参数：", model_args)
    print("数据参数：", data_args)
    print("训练参数：", training_args)

    # ----- initial params setup -------------------------
    # set seed
    transformers.set_seed(training_args.seed)
    # disable caching allocator warmup
    utils.disable_caching_allocator_warmup()
    utils.disable_dataset_progress_bar_except_main()
    if getattr(data_args, "disable_caching", False):
        utils.disable_dataset_caching()

    # ----- tokenizer loading -------------------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----- model loading -------------------------
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_config(
        config,
        torch_dtype=getattr(torch, model_args.dtype),
    )
    model.to(torch.cuda.current_device())

    # ----- data loading -------------------------
    dataset = utils.load_pt_dataset(
        dataset_args=data_args.dataset_args
    )

    dataset = dataset.map(
            functools.partial(
                utils.tokenize_and_group,
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
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        return_tensors="pt",
        padding=True,
        label_pad_token_id=tokenizer.pad_token_id,
        max_length=data_args.max_length,
    )

    # ----- Training --------------------------------------------------------------
    logging.info("Start training...")
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=data_collator
    )

    # ----- Training -------------------------
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)          
    tokenizer.save_pretrained(training_args.output_dir) 

if __name__ == "__main__":
    train()

