import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CLIP_MAX_LENGTH = 77


class Vocabulary:
    """Wrapper around a pretrained CLIP tokenizer.

    The interface (encode / save / load / __len__) is preserved so that the
    rest of the codebase does not need to change. Token IDs are produced by
    the CLIP BPE tokenizer and the attention mask is forwarded to downstream
    cross-modal attention modules.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        from transformers import CLIPTokenizer

        self.model_name = model_name
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

    def __len__(self) -> int:
        return self.tokenizer.vocab_size

    def encode(self, text: str, max_words: int) -> Tuple[torch.Tensor, torch.Tensor]:
        max_length = max(1, min(max_words, CLIP_MAX_LENGTH))
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        token_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0].bool()
        return token_ids, attention_mask

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"model_name": self.model_name}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path) -> "Vocabulary":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["model_name"])


def build_vocab(model_name: str = "openai/clip-vit-base-patch32") -> Vocabulary:
    """Instantiate a CLIP-based vocabulary.

    Kept for backward-compatible import paths. Unlike the previous BiLSTM
    implementation, no manifest scan is required because CLIP ships with a
    fixed vocabulary.
    """

    return Vocabulary(model_name)


class MuralInpaintingDataset(Dataset):
    def __init__(
        self,
        manifest,
        vocab: Vocabulary,
        image_size: int = 256,
        max_words: int = CLIP_MAX_LENGTH,
    ) -> None:
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.vocab = vocab
        self.max_words = max_words
        self.samples = self._read_manifest(self.manifest)

        self.image_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        self.mask_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        image = Image.open(self._resolve(sample["image"])).convert("RGB")
        mask = Image.open(self._resolve(sample["mask"])).convert("L")
        text = sample["text"]

        image_t = self.image_tf(image)
        mask_t = (self.mask_tf(mask) > 0.5).float()
        token_ids, token_mask = self.vocab.encode(text, self.max_words)

        masked = image_t * (1.0 - mask_t)
        return {
            "image": image_t,
            "mask": mask_t,
            "masked": masked,
            "token_ids": token_ids,
            "token_mask": token_mask,
            "text": text,
        }

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute() or p.exists():
            return p
        return self.root / p

    @staticmethod
    def _read_manifest(manifest: Path) -> List[Dict[str, str]]:
        samples: List[Dict[str, str]] = []
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    for key in ("image", "mask", "text"):
                        if key not in item:
                            raise KeyError(f"manifest sample missing key: {key}")
                    samples.append(item)
        if not samples:
            raise ValueError(f"empty manifest: {manifest}")
        return samples
