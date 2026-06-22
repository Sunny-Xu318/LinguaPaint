from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms

from src.data import Vocabulary
from src.model import build_model
from src.utils import save_tensor_image, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="result.png")
    parser.add_argument("--vocab", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-ema", action="store_true", help="Load EMA weights if available.")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help=(
            "Classifier-free guidance scale. Defaults to ``inference.guidance_scale`` "
            "from the checkpoint config. ``1.0`` disables CFG (conditional only); "
            "``0.0`` produces an unconditional output; values >1 push the result "
            "more strongly towards the text description."
        ),
    )
    return parser.parse_args()


def load_inputs(image_path: str, mask_path: str, image_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    image_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    mask_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )
    image = image_tf(Image.open(image_path).convert("RGB")).unsqueeze(0)
    mask = (mask_tf(Image.open(mask_path).convert("L")).unsqueeze(0) > 0.5).float()
    return image, mask


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    device = select_device(args.device or cfg["device"])

    vocab_path = Path(args.vocab) if args.vocab else Path(args.checkpoint).parent.parent / "vocab.json"
    vocab = Vocabulary.load(vocab_path)

    model = build_model(cfg).to(device)
    if args.use_ema and "ema" in ckpt:
        model.load_state_dict(ckpt["ema"], strict=False)
    else:
        model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    image, mask = load_inputs(args.image, args.mask, cfg["data"]["image_size"])
    token_ids, token_mask = vocab.encode(args.text, cfg["data"]["max_words"])

    image = image.to(device)
    mask = mask.to(device)
    masked = image * (1.0 - mask)
    token_ids = token_ids.unsqueeze(0).to(device)
    token_mask = token_mask.unsqueeze(0).to(device)

    if args.guidance_scale is None:
        guidance_scale = float(cfg.get("inference", {}).get("guidance_scale", 1.0))
    else:
        guidance_scale = float(args.guidance_scale)

    with torch.no_grad():
        if guidance_scale == 1.0:
            output = model(masked, image, mask, token_ids, token_mask)
        else:
            output = model.inpaint_with_cfg(
                masked,
                image,
                mask,
                token_ids,
                token_mask,
                guidance_scale=guidance_scale,
            )

    save_tensor_image(output.gen[0], args.output)


if __name__ == "__main__":
    main()
