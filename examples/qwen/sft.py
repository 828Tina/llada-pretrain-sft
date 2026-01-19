import transformers
from dataclasses import dataclass, field
import torch
from peft import get_peft_model, LoraConfig, TaskType
import utils
import os
import argparse,yaml
import accelerate

os.environ["SWANLAB_PROJECT"]="llada-sft"

# ----- Arguments -------------------------

@dataclass
class ModelArguments:
    model_name_or_path: str = None # overwrite this
    dtype: str = "bfloat16"
    # --- fold PEFT args here ---
    lora: bool = False
    target_modules: str = "all-linear"
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    bias: str = "none"
    modules_to_save: str = None


@dataclass
class DataArguments:
    dataset_args: str = None # overwrite this
    disable_caching: bool = False
    max_length: int = 1024
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
    warmup_ratio: float = 0.1
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
    tmp_parser.add_argument("--config_file", required=True)
    tmp_args, _ = tmp_parser.parse_known_args()

    with open(tmp_args.config_file, "r", encoding="utf-8") as f:
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
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=getattr(torch, model_args.dtype),
        trust_remote_code=True,
    )

    # peft
    lora_config = LoraConfig(
        r=model_args.r,
        lora_alpha=model_args.lora_alpha,
        target_modules="all-linear" if model_args.target_modules == "all-linear" else [m.strip() for m in model_args.target_modules.split(",") if m.strip()],
        lora_dropout=model_args.lora_dropout,
        bias=model_args.bias,
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=model_args.modules_to_save.split(",") if model_args.modules_to_save is not None else None,
    )
    if model_args.lora:
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        print("PEFT 模型参数：", model.print_trainable_parameters())
    
    model.to(torch.cuda.current_device())

    # ----- dataset loading and processing -------------------------
    datasets=utils.load_sft_dataset(data_args.dataset_args)
    # ----- data_collator -------------------------
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        return_tensors="pt",
        padding=True,
        label_pad_token_id=tokenizer.pad_token_id,
        max_length=data_args.max_length,
    )

    datasets = utils.post_process_dataset(datasets, data_args)
    # ----- Trainer -------------------------
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ----- Training -------------------------
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)          
    tokenizer.save_pretrained(training_args.output_dir)   

if __name__ == "__main__":
    train()




