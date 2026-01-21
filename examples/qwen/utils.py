import os
import accelerate
from datasets import load_from_disk,DatasetDict,Dataset
import torch
from itertools import chain
import re
import datasets

# ----- initial params setup -------------------------
def disable_caching_allocator_warmup():
    try:
        from transformers import modeling_utils as _mu

        def _noop(*args, **kwargs):
            return

        _mu.caching_allocator_warmup = _noop
    except Exception:
        pass


def disable_dataset_progress_bar_except_main():
    # state = accelerate.PartialState()  # figures out your rank/world automatically
    from datasets.utils.logging import disable_progress_bar, enable_progress_bar

    if accelerate.PartialState().is_main_process:
        enable_progress_bar()
    else:
        disable_progress_bar()

def disable_dataset_caching():
    from datasets import disable_caching

    disable_caching()
    tmp_root = f"/tmp/hf_cache_rank{accelerate.PartialState().process_index}"
    os.environ["HF_DATASETS_CACHE"] = tmp_root
    os.environ["HF_DATASETS_TEMP_DIR"] = tmp_root
    os.makedirs(tmp_root, exist_ok=True)

# ------------------------- PEFT -------------------------------
# ----- dataset loading and processing -------------------------
def parse_spec(spec: str):
    """
    Parse a general 'name[a:b,c:d]' or 'a=b,c=d' style specification.

    Supports:
      - Bare name, e.g. "foo/bar"
      - Optional bracket suffix with comma-separated entries:
          key:value or key:int_value (underscores allowed)
      - Optional "key=value" pairs outside the bracket.

    Returns:
      name: str or None
      kv_dict: dict of key/value pairs (all combined)
    """

    def _parse_kv_string(s: str) -> dict:
        """Parse comma-separated key=value pairs, e.g. 'a=1,b=2'."""
        return dict(part.split("=", 1) for part in s.split(",") if "=" in part)

    s = spec.strip()

    # Extract bracket content if present
    m = re.search(r"\[(.*?)\]$", s)
    bracket_kvs = {}
    numeric_kvs = {}
    if m:
        bracket = m.group(1).strip()
        if bracket:
            for part in bracket.split(","):
                part = part.strip()
                if not part:
                    continue
                if ":" not in part:
                    raise ValueError(
                        f"Invalid entry '{part}' in '{spec}' (expected key:value)."
                    )
                key, value = part.split(":", 1)
                key = key.strip()
                value = value.strip()

                # Integers (with optional underscores)
                if re.fullmatch(r"\d(?:_?\d)*", value):
                    numeric_kvs[key] = int(value.replace("_", ""))
                else:
                    bracket_kvs[key] = value

        # Remove the bracket suffix from the working string
        s = s[: m.start()].rstrip()

    # Determine name (if any) and parse outer kvs (if any)
    name = None
    if "=" in s:
        kv_dict = dict(_parse_kv_string(s))
    else:
        kv_dict = {}
        if s:
            name = s  # could represent a dataset, resource, or identifier

    # Merge: bracket options and numeric keys last
    kv_dict.update(bracket_kvs)
    kv_dict.update(numeric_kvs)

    return name, kv_dict

def load_sft_dataset(dataset_args:str)->DatasetDict:
    # split dataset_args
    specs = [p.strip() for p in re.split(r"[|+]", dataset_args) if p.strip()]
    name,kvs=parse_spec(specs[0])

    # DatasetDict
    ds = load_from_disk(name)

    # split train/test if specified
    train_datasets=ds.select(range(kvs.get("train",len(ds))))
    test_datasets=ds.select(range(kvs.get("train",len(ds)),kvs.get("train",len(ds))+kvs.get("test",len(ds))))

    return DatasetDict({"train":train_datasets,"test":test_datasets})

def clip_row(row: dict, max_length: int, truncation: str = "right") -> dict:
    for key in ("input_ids", "labels", "attention_mask"):
        if key in row:
            if truncation == "right":
                row[key] = row[key][:max_length]
            elif truncation == "left":
                row[key] = row[key][-max_length:]
            else:
                raise NotImplementedError
    return row

def post_process_dataset(
    dataset: datasets.DatasetDict, data_args
) -> datasets.DatasetDict:
    """
    Post-process dataset by filtering or truncating sequences.

    Args:
        dataset: Dataset dictionary to process.
        data_args: Data arguments with max_length and truncation settings.

    Returns:
        Processed dataset dictionary.
    """
    if data_args.truncation == "filter":
        return dataset.filter(
            lambda row: len(row["input_ids"]) <= data_args.max_length,
            num_proc=data_args.num_proc,
            desc=f"Filtering samples with length <= {data_args.max_length}",
        )
    elif data_args.truncation == "right":
        # do this only if dataset has "prompt_len"
        if "prompt_len" in dataset.column_names["train"]:
            dataset = dataset.filter(
                lambda row: row["prompt_len"] <= data_args.max_length,
                num_proc=data_args.num_proc,
                desc=f"Filtering samples with `prompt_len` <= {data_args.max_length}",
            )
        return dataset.map(
            lambda row: clip_row(row, data_args.max_length, truncation="right"),
            num_proc=data_args.num_proc,
            desc=f"Right-truncating samples to max_length={data_args.max_length}",
        )
    else:
        raise NotImplementedError



# ------------------------- Pretrain ---------------------------
# ----- dataset loading and processing -------------------------
def load_pt_dataset(dataset_args:str)->DatasetDict:
    # split dataset_args
    specs = [p.strip() for p in re.split(r"[|+]", dataset_args) if p.strip()]
    name,kvs=parse_spec(specs[0])

    # DatasetDict
    ds = load_from_disk(name)

    # split train/test if specified
    train_datasets=ds.select(range(kvs.get("train",len(ds))))
    test_datasets=ds.select(range(kvs.get("train",len(ds)),kvs.get("train",len(ds))+kvs.get("test",len(ds))))

    return DatasetDict({"train":train_datasets,"test":test_datasets})

def tokenize_and_group(
    examples,
    tokenizer,
    text_field: str = "text",
    seq_length: int = 1024,
    insert_eos: bool = False,
    drop_tail: bool = True,
    add_special_tokens: bool = False,
):
    """
    Tokenize text examples and group into fixed-length sequences.

    Concatenates all tokenized text and splits into chunks of seq_length.
    Optionally drops incomplete trailing chunks.

    Args:
        examples: Batch of examples with text field.
        tokenizer: Tokenizer to use.
        text_field: Name of the text field in examples.
        seq_length: Target sequence length for chunks.
        insert_eos: If True, append EOS token to each text sample.
        drop_tail: If True, drop incomplete final chunk; if False, keep it.
        add_special_tokens: Whether to add special tokens during tokenization.

    Returns:
        Dictionary with input_ids and labels as lists of token sequences.
    """
    # 1) Tokenize (batched input)
    tokenized = tokenizer(examples[text_field], add_special_tokens=add_special_tokens)
    ids = tokenized["input_ids"]

    # --- optionally append EOS to each sample ---
    if insert_eos:
        eos_id = getattr(tokenizer, "eos_token_id")
        assert eos_id
        # append EOS only if the sample doesn't already end with it
        ids = [seq + ([] if (seq and seq[-1] == eos_id) else [eos_id]) for seq in ids]
    # ----------------------------------------------------------------

    # 2) Flatten and concatenate all token lists
    concatenated = list(chain.from_iterable(ids))
    if not concatenated:
        return {"input_ids": [], "labels": []}  # Safe return for empty batch

    # 3) Calculate the total length based on drop_tail
    if drop_tail:
        total_len = (len(concatenated) // seq_length) * seq_length
        concatenated = concatenated[:total_len]  # Truncate the last incomplete chunk
    else:
        total_len = len(concatenated)

    # Split into fixed-length chunks
    chunks = [concatenated[i : i + seq_length] for i in range(0, total_len, seq_length)]

    return {
        "input_ids": chunks,
        "labels": [c[:] for c in chunks],  # Labels are the same as input_ids
    }


