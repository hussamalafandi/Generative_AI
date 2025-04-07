import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from datasets import Dataset, load_dataset


def tokenize_function(examples, tokenizer, return_special_tokens_mask=True):
    """
    Tokenizes a batch of text examples using the specified tokenizer.

    Args:
        examples (dict): A batch of examples with a "text" field.
        tokenizer (PreTrainedTokenizer): A Hugging Face tokenizer instance.
        return_special_tokens_mask (bool): Whether to return special token masks.

    Returns:
        dict: Tokenized output with input_ids and optional special token masks.
    """
    return tokenizer(
        examples["text"],
        return_special_tokens_mask=return_special_tokens_mask,
        truncation=False,
        add_special_tokens=True,
    )


def group_texts(examples, block_size):
    """
    Groups tokenized texts into fixed-size blocks for language modeling.

    Args:
        examples (dict): A batch of tokenized examples (lists of input_ids).
        block_size (int): The length of each fixed-size input sequence.

    Returns:
        dict: A dictionary with input_ids (and possibly other fields) divided into blocks.
    """
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = (len(concatenated["input_ids"]) // block_size) * block_size
    result = {
        k: [concatenated[k][i:i + block_size]
            for i in range(0, total_length, block_size)]
        for k in concatenated.keys()
    }
    return result


def create_dataloader(
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
    """
    Creates a PyTorch DataLoader for language modeling using Hugging Face datasets and tokenizers.

    Args:
        dataset_name (str or Dataset): Name of the Hugging Face dataset or a preloaded Dataset object.
        tokenizer_name (str): Hugging Face tokenizer name or path (e.g., "gpt2").
        block_size (int): Sequence length for each training example.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        mlm (bool): If True, uses masked language modeling (BERT-style); otherwise, causal LM.
        split (str): Dataset split to use (e.g., "train", "validation").
        streaming (bool): If True, loads the dataset in streaming mode.

    Returns:
        DataLoader: A PyTorch DataLoader instance for the language modeling task.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    # Tokenize dataset
    tokenized = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Group texts into blocks
    grouped = tokenized.map(
        lambda x: group_texts(x, block_size),
        batched=True,
    )

    # Use collator for padding and label masking
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=mlm)

    # Return PyTorch DataLoader
    return DataLoader(
        grouped,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True
    )
