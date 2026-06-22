from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import MuralInpaintingDataset, Vocabulary
from src.losses import LPIPSLoss, generator_losses, hinge_d_loss
from src.model import ClipImageEncoder, SNPatchDiscriminator, build_model
from src.utils import (
    AverageMeter,
    ModelEMA,
    clip_gradients,
    load_config,
    save_tensor_image,
    select_device,
    set_requires_grad,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = select_device(cfg["device"])

    out_dir = Path(cfg["train"]["out_dir"])
    ckpt_dir = out_dir / "checkpoints"
    sample_dir = out_dir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = out_dir / "vocab.json"
    if args.resume and vocab_path.exists():
        vocab = Vocabulary.load(vocab_path)
    else:
        vocab = Vocabulary(cfg["model"]["clip_model_name"])
        vocab.save(vocab_path)

    train_set = MuralInpaintingDataset(
        cfg["data"]["train_manifest"],
        vocab,
        image_size=cfg["data"]["image_size"],
        max_words=cfg["data"]["max_words"],
    )
    loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    model = build_model(cfg).to(device)
    d_patch = SNPatchDiscriminator(cfg["model"]["base_channels"]).to(device)
    damsm_encoder = ClipImageEncoder(
        clip_model_name=cfg["model"]["clip_model_name"],
        feat_dim=cfg["model"]["text_hidden_dim"],
    ).to(device)

    lpips_loss_fn = None
    if cfg["loss"].get("lambda_lpips", 0.0) > 0.0:
        lpips_loss_fn = LPIPSLoss(net=cfg["loss"].get("lpips_net", "vgg")).to(device)

    g_params = list(model.parameters()) + list(damsm_encoder.parameters())
    opt_g = torch.optim.Adam(
        g_params,
        lr=cfg["train"]["lr_g"],
        betas=tuple(cfg["train"]["betas"]),
    )
    opt_d = torch.optim.Adam(
        list(d_patch.parameters()),
        lr=cfg["train"]["lr_d"],
        betas=tuple(cfg["train"]["betas"]),
    )

    ema = None
    if cfg["train"].get("ema_decay", 0.0) > 0:
        ema = ModelEMA(model, decay=cfg["train"]["ema_decay"])

    grad_clip = cfg["train"].get("grad_clip", 0.0)
    cfg_dropout = float(cfg["train"].get("cfg_dropout", 0.0))

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        d_patch.load_state_dict(ckpt["d_patch"])
        damsm_encoder.load_state_dict(ckpt["damsm"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        if ema is not None and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        start_epoch = ckpt["epoch"] + 1

    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        model.train()
        damsm_encoder.train()
        d_patch.train()

        meter = AverageMeter()
        pbar = tqdm(loader, desc=f"epoch {epoch}")

        for step, batch in enumerate(pbar, start=1):
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            masked = batch["masked"].to(device)
            token_ids = batch["token_ids"].to(device)
            token_mask = batch["token_mask"].to(device)

            output = model(
                masked,
                image,
                mask,
                token_ids,
                token_mask,
                cfg_dropout=cfg_dropout,
            )

            set_requires_grad(d_patch, True)
            opt_d.zero_grad(set_to_none=True)
            real_patch = d_patch(image)
            fake_patch = d_patch(output.gen.detach())
            loss_d = hinge_d_loss(real_patch, fake_patch)
            loss_d.backward()
            opt_d.step()

            set_requires_grad(d_patch, False)
            opt_g.zero_grad(set_to_none=True)
            patch_fake_logits = d_patch(output.gen)
            loss_g, logs = generator_losses(
                output,
                image,
                mask,
                token_mask,
                patch_fake_logits,
                damsm_encoder,
                cfg["loss"],
                lpips_loss_fn=lpips_loss_fn,
            )
            loss_g.backward()
            if grad_clip > 0:
                clip_gradients(g_params, grad_clip)
            opt_g.step()

            if ema is not None:
                ema.update(model)

            logs["loss_d"] = loss_d.detach()
            meter.update(logs)
            if step % cfg["train"]["log_every"] == 0:
                pbar.set_postfix({k: f"{v:.4f}" for k, v in meter.summary().items()})

        with torch.no_grad():
            grid = torch.cat([masked[:4], output.gen[:4], image[:4]], dim=0)
            save_tensor_image(grid, sample_dir / f"epoch_{epoch:04d}.png")

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "d_patch": d_patch.state_dict(),
            "damsm": damsm_encoder.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "config": cfg,
        }
        if ema is not None:
            state["ema"] = ema.state_dict()
        save_checkpoint(ckpt_dir / "latest.pt", state)
        if epoch % cfg["train"]["save_every"] == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch:04d}.pt", state)


if __name__ == "__main__":
    main()
