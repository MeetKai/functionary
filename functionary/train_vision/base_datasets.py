from torch.utils.data import Dataset
from typing import Any, List, Dict
from functionary.prompt_template import prompt_utils, get_prompt_template_from_tokenizer
from PIL import Image


class VisionDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data: List[Dict],
        tokenizer: Any,
        pretrained_path: str,
        pad_img_path: str,
        max_length: int,
        use_img_pad_token: bool = True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.pretrained_path = pretrained_path
        self.raw_data = raw_data
        self.pad_img = None
        self.pad_img_path = pad_img_path
        self.prompt_template = get_prompt_template_from_tokenizer(tokenizer)
        self.max_length = max_length # tokenizer.model_max_length
        self.pad_img = Image.open(open(pad_img_path, "rb"))
        self.use_img_pad_token = use_img_pad_token


class CustomCollator:
    def __init__(self, tokenizer: Any, model: Any) -> None:
        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, features: Any) -> Any:
        raise NotImplemented