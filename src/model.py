from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def conv_block(in_ch: int, out_ch: int, stride: int = 1, norm: bool = True) -> nn.Sequential:
    layers: List[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
    ]
    if norm:
        layers.append(nn.InstanceNorm2d(out_ch, affine=True))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def up_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.ReLU(inplace=True),
    )


class TextEncoder(nn.Module):
    """Frozen CLIP text encoder with learnable projections.

    Wraps a pretrained CLIP text model from HuggingFace transformers, keeps
    the CLIP weights frozen, and projects the per-token / pooled features to
    the model's shared semantic dimension via small linear heads.

    Also pre-tokenises an empty prompt and registers it as a buffer so that
    classifier-free guidance can swap in a null text representation without
    requiring the tokenizer at runtime.
    """

    def __init__(
        self,
        clip_model_name: str,
        hidden_dim: int,
        max_length: int = 77,
    ) -> None:
        super().__init__()
        from transformers import CLIPTextModel, CLIPTokenizer

        self.clip = CLIPTextModel.from_pretrained(clip_model_name)
        for p in self.clip.parameters():
            p.requires_grad_(False)
        self.clip.eval()

        clip_dim = self.clip.config.hidden_size
        self.word_proj = nn.Linear(clip_dim, hidden_dim)
        self.sent_proj = nn.Linear(clip_dim, hidden_dim)

        tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        null = tokenizer(
            "",
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.register_buffer("null_token_ids", null["input_ids"][0].long(), persistent=False)
        self.register_buffer(
            "null_token_mask", null["attention_mask"][0].bool(), persistent=False
        )

    def train(self, mode: bool = True) -> "TextEncoder":
        super().train(mode)
        self.clip.eval()
        return self

    def get_null_text(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = self.null_token_ids.unsqueeze(0).expand(batch_size, -1).contiguous()
        mask = self.null_token_mask.unsqueeze(0).expand(batch_size, -1).contiguous()
        return ids, mask

    def forward(
        self, token_ids: torch.Tensor, token_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            outputs = self.clip(
                input_ids=token_ids,
                attention_mask=token_mask.long(),
                return_dict=True,
            )
            last_hidden = outputs.last_hidden_state
            pooled = outputs.pooler_output

        words = self.word_proj(last_hidden)
        words = words * token_mask.unsqueeze(-1).float()
        sentence = self.sent_proj(pooled)
        return words, sentence


class ConditionalAugmentation(nn.Module):
    def __init__(self, in_dim: int, ca_dim: int) -> None:
        super().__init__()
        self.mu = nn.Linear(in_dim, ca_dim)
        self.logvar = nn.Linear(in_dim, ca_dim)

    def forward(self, sentence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.mu(sentence)
        logvar = self.logvar(sentence).clamp(-8.0, 8.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps, mu, logvar


class ImageEncoder(nn.Module):
    """U-Net style encoder with three skip connections and a deep bottleneck.

    The extra ``stride=2`` layer at the end produces a low-resolution feature
    map (``image_size / 8``) that is small enough for the bottleneck self- and
    cross-attention transformer blocks to operate on.
    """

    def __init__(self, base_channels: int) -> None:
        super().__init__()
        c = base_channels
        # 256 -> 256
        self.layer0 = conv_block(4, c, stride=1, norm=False)
        # 256 -> 128
        self.layer1 = conv_block(c, c * 2, stride=2)
        # 128 -> 64
        self.layer2 = conv_block(c * 2, c * 4, stride=2)
        # 64  -> 32
        self.layer3 = conv_block(c * 4, c * 8, stride=2)
        # bottleneck refinement (no spatial change)
        self.bottleneck = nn.Sequential(
            conv_block(c * 8, c * 8),
            conv_block(c * 8, c * 8),
        )
        self.out_channels = c * 8
        self.skip_channels: Tuple[int, int, int] = (c, c * 2, c * 4)

    def forward(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        x = torch.cat([image, mask], dim=1)
        s0 = self.layer0(x)
        s1 = self.layer1(s0)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        out = self.bottleneck(s3)
        return out, (s0, s1, s2)


class MultiHeadAttention(nn.Module):
    """Light-weight multi-head attention built on top of SDPA.

    Supports both self-attention (``kv`` = ``q``) and cross-attention with a
    differently sized key/value tensor via ``kv_dim``. ``kv_mask`` is an
    optional ``[B, n_kv]`` boolean mask where ``True`` indicates valid tokens.
    """

    def __init__(self, query_dim: int, num_heads: int, kv_dim: Optional[int] = None) -> None:
        super().__init__()
        kv_dim = kv_dim if kv_dim is not None else query_dim
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads

        self.q_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim, query_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, query_dim, bias=False)
        self.out_proj = nn.Linear(query_dim, query_dim)

    def forward(
        self,
        q: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if kv is None:
            kv = q
        b, n_q, _ = q.shape
        n_kv = kv.shape[1]

        q_h = self.q_proj(q).reshape(b, n_q, self.num_heads, self.head_dim).transpose(1, 2)
        k_h = self.k_proj(kv).reshape(b, n_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v_h = self.v_proj(kv).reshape(b, n_kv, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if kv_mask is not None:
            attn_mask = kv_mask[:, None, None, :].expand(b, self.num_heads, n_q, n_kv)
        out = F.scaled_dot_product_attention(q_h, k_h, v_h, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, n_q, self.num_heads * self.head_dim)
        return self.out_proj(out)


class TransformerBottleneckBlock(nn.Module):
    """Self-attn + text cross-attn + FFN block, applied to the U-Net bottleneck.

    Operates on a flattened ``[B, H*W, C]`` token sequence so that gradients
    flow through standard SDPA. Pre-norm residual structure mirrors modern
    diffusion U-Nets (Stable Diffusion, ControlNet) and provides a single,
    clean place to inject text conditioning into the generator.
    """

    def __init__(self, channels: int, text_dim: int, num_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.self_attn = MultiHeadAttention(channels, num_heads)

        self.norm2 = nn.LayerNorm(channels)
        self.cross_attn = MultiHeadAttention(channels, num_heads, kv_dim=text_dim)

        self.norm3 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * mlp_ratio),
            nn.GELU(),
            nn.Linear(channels * mlp_ratio, channels),
        )

    def forward(
        self,
        feat: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = feat.shape
        x = feat.flatten(2).transpose(1, 2)

        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), text, kv_mask=text_mask)
        x = x + self.ffn(self.norm3(x))

        return x.transpose(1, 2).reshape(b, c, h, w)


class TransformerBottleneck(nn.Module):
    """Stack of ``TransformerBottleneckBlock`` modules with a final 1x1 conv."""

    def __init__(self, channels: int, text_dim: int, num_heads: int, num_blocks: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBottleneckBlock(channels, text_dim, num_heads) for _ in range(num_blocks)]
        )
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(
        self,
        feat: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = feat
        for block in self.blocks:
            x = block(x, text, text_mask)
        return feat + self.proj_out(x)


class Decoder(nn.Module):
    """U-Net style decoder consuming three encoder skip features."""

    def __init__(
        self,
        in_ch: int,
        base_channels: int,
        skip_channels: Tuple[int, int, int],
    ) -> None:
        super().__init__()
        c = base_channels
        s0_c, s1_c, s2_c = skip_channels
        self.proj = conv_block(in_ch, c * 8)
        # 32 -> 64
        self.up1 = up_block(c * 8, c * 4)
        self.fuse1 = conv_block(c * 4 + s2_c, c * 4)
        # 64 -> 128
        self.up2 = up_block(c * 4, c * 2)
        self.fuse2 = conv_block(c * 2 + s1_c, c * 2)
        # 128 -> 256
        self.up3 = up_block(c * 2, c)
        self.fuse3 = conv_block(c + s0_c, c)
        self.head = nn.Sequential(
            nn.Conv2d(c, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(
        self,
        x: torch.Tensor,
        skips: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        s0, s1, s2 = skips
        x = self.proj(x)
        x = self.up1(x)
        x = self.fuse1(torch.cat([x, s2], dim=1))
        x = self.up2(x)
        x = self.fuse2(torch.cat([x, s1], dim=1))
        x = self.up3(x)
        x = self.fuse3(torch.cat([x, s0], dim=1))
        return self.head(x)


@dataclass
class ModelOutput:
    gen: torch.Tensor
    raw: torch.Tensor
    ca_mu: torch.Tensor
    ca_logvar: torch.Tensor
    words: torch.Tensor
    sentence: torch.Tensor


class TextGuidedInpainter(nn.Module):
    """Single-path text-conditioned U-Net inpainter.

    Replaces the original variational dual-path design with a ControlNet /
    Stable-Diffusion-style architecture: a single U-Net encoder produces a
    deep bottleneck feature, transformer blocks at the bottleneck inject text
    guidance through cross-attention, and a U-Net decoder reconstructs the
    image. Classifier-free guidance is supported through a null-text buffer
    in the text encoder; ``cfg_dropout`` randomly replaces a sample's text
    with the null prompt during training, and ``inpaint_with_cfg`` blends
    conditional and unconditional outputs at inference time.
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        text_hidden_dim: int = 256,
        base_channels: int = 64,
        ca_dim: int = 256,
        attn_heads: int = 4,
        num_bottleneck_blocks: int = 2,
        max_words: int = 77,
    ) -> None:
        super().__init__()
        self.text_encoder = TextEncoder(clip_model_name, text_hidden_dim, max_length=max_words)
        self.ca = ConditionalAugmentation(text_hidden_dim, ca_dim)
        self.image_encoder = ImageEncoder(base_channels)
        self.sent_token_proj = nn.Linear(ca_dim, text_hidden_dim)

        self.bottleneck_attn = TransformerBottleneck(
            channels=self.image_encoder.out_channels,
            text_dim=text_hidden_dim,
            num_heads=attn_heads,
            num_blocks=num_bottleneck_blocks,
        )
        self.decoder = Decoder(
            self.image_encoder.out_channels,
            base_channels,
            self.image_encoder.skip_channels,
        )

    def _resolve_text(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        cfg_dropout: float,
        force_null_text: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = token_ids.size(0)
        if force_null_text:
            null_ids, null_mask = self.text_encoder.get_null_text(b)
            return null_ids.to(token_ids.device), null_mask.to(token_mask.device)

        if not self.training or cfg_dropout <= 0.0:
            return token_ids, token_mask

        drop = torch.rand(b, device=token_ids.device) < cfg_dropout
        if not drop.any():
            return token_ids, token_mask

        null_ids, null_mask = self.text_encoder.get_null_text(b)
        null_ids = null_ids.to(token_ids.device)
        null_mask = null_mask.to(token_mask.device)
        token_ids = torch.where(drop[:, None], null_ids, token_ids)
        token_mask = torch.where(drop[:, None], null_mask, token_mask)
        return token_ids, token_mask

    def _compose_text_features(
        self,
        words: torch.Tensor,
        sentence_ca: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = words.size(0)
        sent_token = self.sent_token_proj(sentence_ca).unsqueeze(1)
        text_features = torch.cat([sent_token, words], dim=1)
        sent_mask = token_mask.new_ones(b, 1, dtype=torch.bool)
        text_mask = torch.cat([sent_mask, token_mask.bool()], dim=1)
        return text_features, text_mask

    def forward(
        self,
        masked: torch.Tensor,
        image: torch.Tensor,
        mask: torch.Tensor,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        inference: bool = False,
        cfg_dropout: float = 0.0,
        force_null_text: bool = False,
    ) -> ModelOutput:
        del inference  # kept for backward compatibility; single-path forward.

        token_ids, token_mask = self._resolve_text(
            token_ids, token_mask, cfg_dropout, force_null_text
        )

        words, sentence = self.text_encoder(token_ids, token_mask)
        sentence_ca, ca_mu, ca_logvar = self.ca(sentence)
        text_features, text_mask = self._compose_text_features(words, sentence_ca, token_mask)

        feat, skips = self.image_encoder(masked, mask)
        feat = self.bottleneck_attn(feat, text_features, text_mask)
        raw = self.decoder(feat, skips)
        gen = image * (1.0 - mask) + raw * mask

        return ModelOutput(
            gen=gen,
            raw=raw,
            ca_mu=ca_mu,
            ca_logvar=ca_logvar,
            words=words,
            sentence=sentence,
        )

    @torch.no_grad()
    def inpaint_with_cfg(
        self,
        masked: torch.Tensor,
        image: torch.Tensor,
        mask: torch.Tensor,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        guidance_scale: float = 3.0,
    ) -> ModelOutput:
        """Run conditional + unconditional forwards and combine via CFG.

        ``guidance_scale = 1.0`` is equivalent to the conditional forward.
        ``guidance_scale = 0.0`` is the unconditional forward. Higher scales
        push the generation more aggressively towards the text description.
        """

        cond = self.forward(masked, image, mask, token_ids, token_mask)
        if guidance_scale == 1.0:
            return cond

        uncond = self.forward(
            masked, image, mask, token_ids, token_mask, force_null_text=True
        )
        # Combining the composited outputs leaves the unmasked region
        # untouched (background cancels out in the difference).
        gen = uncond.gen + guidance_scale * (cond.gen - uncond.gen)
        gen = image * (1.0 - mask) + gen * mask  # numerical safety
        return ModelOutput(
            gen=gen,
            raw=cond.raw,
            ca_mu=cond.ca_mu,
            ca_logvar=cond.ca_logvar,
            words=cond.words,
            sentence=cond.sentence,
        )


# Backward compatible alias for older imports / checkpoints.
TextGuidedMuralInpainter = TextGuidedInpainter


class SNPatchDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(3, c, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(c, c * 2, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(c * 4, 1, 3, padding=1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClipImageEncoder(nn.Module):
    """Frozen CLIP vision encoder with learnable projections.

    Used as the image branch of the DAMSM-style image-text matching loss.
    The pretrained CLIP visual transformer is kept frozen; only the small
    projection layers are trained so that the patch and global features
    align with the project's shared semantic dimension.

    The encoder accepts images normalised with ``mean=0.5, std=0.5`` (the
    convention used by the training/inference pipeline) and internally
    re-normalises them to CLIP's expected statistics.
    """

    IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
    IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, clip_model_name: str, feat_dim: int) -> None:
        super().__init__()
        from transformers import CLIPVisionModel

        self.clip = CLIPVisionModel.from_pretrained(clip_model_name)
        for p in self.clip.parameters():
            p.requires_grad_(False)
        self.clip.eval()

        clip_dim = self.clip.config.hidden_size
        self.input_size = self.clip.config.image_size

        self.local_proj = nn.Linear(clip_dim, feat_dim)
        self.global_proj = nn.Linear(clip_dim, feat_dim)

        clip_mean = torch.tensor(self.IMAGE_MEAN).view(1, 3, 1, 1)
        clip_std = torch.tensor(self.IMAGE_STD).view(1, 3, 1, 1)
        self.register_buffer("input_offset", 2.0 * clip_mean - 1.0)
        self.register_buffer("input_scale", 2.0 * clip_std)

    def train(self, mode: bool = True) -> "ClipImageEncoder":
        super().train(mode)
        self.clip.eval()
        return self

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if image.shape[-1] != self.input_size or image.shape[-2] != self.input_size:
            image = F.interpolate(
                image,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )
        normalised = (image - self.input_offset) / self.input_scale

        # NOTE: do not use ``torch.no_grad`` here. The input image typically
        # comes from the generator and must receive gradients for the DAMSM
        # loss to update the generator. CLIP weights stay frozen because their
        # ``requires_grad`` flags are False.
        outputs = self.clip(pixel_values=normalised, return_dict=True)
        last_hidden = outputs.last_hidden_state
        pooled = outputs.pooler_output

        patch_features = last_hidden[:, 1:, :]
        local = self.local_proj(patch_features)
        global_feat = self.global_proj(pooled)

        b, n, d = local.shape
        side = int(n**0.5)
        if side * side != n:
            local_feat = local.transpose(1, 2)
        else:
            local_feat = local.transpose(1, 2).reshape(b, d, side, side)
        return local_feat, global_feat


def build_model(config: Dict) -> TextGuidedInpainter:
    m = config["model"]
    return TextGuidedInpainter(
        clip_model_name=m["clip_model_name"],
        text_hidden_dim=m["text_hidden_dim"],
        base_channels=m["base_channels"],
        ca_dim=m["ca_dim"],
        attn_heads=m["attn_heads"],
        num_bottleneck_blocks=m.get("num_bottleneck_blocks", 2),
        max_words=config.get("data", {}).get("max_words", 77),
    )
