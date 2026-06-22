from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ClipImageEncoder, ModelOutput


def kl_standard(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def hinge_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def hinge_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


class LPIPSLoss(nn.Module):
    """Thin wrapper around the official ``lpips`` package.

    Inputs are expected to be in ``[-1, 1]`` (matching the rest of the
    pipeline). Internally calls ``lpips.LPIPS`` with the requested backbone
    and freezes its parameters so they never appear in the optimiser.
    """

    def __init__(self, net: str = "vgg") -> None:
        super().__init__()
        import lpips as _lpips

        self.lpips = _lpips.LPIPS(net=net, verbose=False)
        for p in self.lpips.parameters():
            p.requires_grad_(False)
        self.lpips.eval()

    def train(self, mode: bool = True) -> "LPIPSLoss":
        super().train(mode)
        self.lpips.eval()
        return self

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.lpips(pred, target).mean()


def damsm_pair_loss(
    image: torch.Tensor,
    words: torch.Tensor,
    sentence: torch.Tensor,
    token_mask: torch.Tensor,
    image_encoder: ClipImageEncoder,
    gamma1: float = 4.0,
    gamma2: float = 5.0,
    gamma3: float = 10.0,
) -> torch.Tensor:
    """DAMSM-style image-text matching loss using a frozen CLIP image encoder.

    Computes both word-image and sentence-image cross-entropy contrastive losses
    over the batch (each image must match its paired text against all other
    texts in the batch). The image features come from a pretrained CLIP visual
    transformer; only the projection layers and the text encoder gradients
    flow through the loss.
    """

    b = image.size(0)
    if b < 2:
        return image.new_zeros(())

    local_feat, global_feat = image_encoder(image)
    patches = local_feat.flatten(2).transpose(1, 2)

    patches = F.normalize(patches, dim=-1)
    words_n = F.normalize(words, dim=-1)
    global_n = F.normalize(global_feat, dim=-1)
    sentence_n = F.normalize(sentence, dim=-1)

    word_valid = token_mask.float()

    sim_all = torch.einsum("ihd,jld->ijhl", patches, words_n)
    sim_all = sim_all * word_valid[None, :, None, :]

    alpha = (gamma1 * sim_all).softmax(dim=2)
    word_ctx = torch.einsum("ijhl,ihd->ijld", alpha, patches)

    words_n_expanded = words_n.unsqueeze(0).expand(b, -1, -1, -1)
    word_sim = (word_ctx * words_n_expanded).sum(dim=-1)

    word_sim = word_sim.masked_fill(word_valid[None, :, :] < 0.5, -1e4)
    r_score = torch.logsumexp(gamma2 * word_sim, dim=-1) / gamma2

    labels = torch.arange(b, device=image.device)

    word_logits = gamma3 * r_score
    word_loss = 0.5 * (
        F.cross_entropy(word_logits, labels) + F.cross_entropy(word_logits.t(), labels)
    )

    sent_logits = gamma3 * torch.matmul(global_n, sentence_n.transpose(0, 1))
    sent_loss = 0.5 * (
        F.cross_entropy(sent_logits, labels) + F.cross_entropy(sent_logits.t(), labels)
    )

    return word_loss + sent_loss


def damsm_total_loss(
    output: ModelOutput,
    image: torch.Tensor,
    token_mask: torch.Tensor,
    image_encoder: ClipImageEncoder,
) -> torch.Tensor:
    """DAMSM losses on the generated image and the ground-truth image."""

    loss = damsm_pair_loss(output.gen, output.words, output.sentence, token_mask, image_encoder)
    loss = loss + damsm_pair_loss(image, output.words, output.sentence, token_mask, image_encoder)
    return loss


def generator_losses(
    output: ModelOutput,
    image: torch.Tensor,
    mask: torch.Tensor,
    token_mask: torch.Tensor,
    patch_fake_logits: torch.Tensor,
    damsm_encoder: ClipImageEncoder,
    weights: Dict[str, float],
    lpips_loss_fn: Optional[LPIPSLoss] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Generator-side loss aggregation for the single-path ControlNet design."""

    ca_kl = kl_standard(output.ca_mu, output.ca_logvar)

    valid = mask.sum().clamp(min=1.0)
    app = (output.gen - image).abs().sum() / (valid * image.size(1))

    damsm = damsm_total_loss(output, image, token_mask, damsm_encoder)

    adv = hinge_g_loss(patch_fake_logits)

    if lpips_loss_fn is not None and weights.get("lambda_lpips", 0.0) > 0.0:
        lpips_val = lpips_loss_fn(output.gen, image)
    else:
        lpips_val = output.gen.new_zeros(())

    total = (
        weights.get("lambda_kl", 0.0) * ca_kl
        + weights.get("lambda_app", 1.0) * app
        + weights.get("lambda_lpips", 0.0) * lpips_val
        + weights.get("lambda_damsm", 0.0) * damsm
        + weights.get("lambda_adv", 0.0) * adv
    )
    logs = {
        "loss_g": total.detach(),
        "kl": ca_kl.detach(),
        "app": app.detach(),
        "lpips": lpips_val.detach(),
        "damsm": damsm.detach(),
        "adv": adv.detach(),
    }
    return total, logs
