import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from datasets import Dataset, load_dataset


def tokenize_function(examples, tokenizer, return_special_tokens_mask=True):
    return tokenizer(
        examples["text"],
        return_special_tokens_mask=return_special_tokens_mask,
        truncation=False,
        add_special_tokens=True,
    )


def group_texts(examples, block_size):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = (len(concatenated["input_ids"]) // block_size) * block_size
    result = {
        k: [concatenated[k][i:i + block_size]
            for i in range(0, total_length, block_size)]
        for k in concatenated.keys()
    }
    return result


def get_text_dataloader(
    dataset_name: str,
    tokenizer_name: str = "gpt2",
    block_size: int = 128,
    batch_size: int = 8,
    shuffle: bool = True,
    mlm: bool = False,
    split: str = "train",
    streaming: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)


    # Tokenize
    tokenized = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Group into blocks
    grouped = tokenized.map(
        lambda x: group_texts(x, block_size),
        batched=True,
    )

    # Create collator
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=mlm)

    # Return DataLoader
    return DataLoader(
        grouped,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True
    )
