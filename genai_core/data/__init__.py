from genai_core.data.image_loader import get_image_dataloader
from genai_core.data.text_loader import get_text_dataloader

def get_dataloader(config):
    if config.task == "image":
        return get_image_dataloader(config)
    elif config.task == "text":
        return get_text_dataloader(config)
    else:
        raise ValueError(f"Unsupported task: {config.task}")
