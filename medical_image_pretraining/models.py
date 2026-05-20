import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)


class GELU(nn.Module):
    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PatchEmbed(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=256):
        super(PatchEmbed, self).__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, images):
        x = self.proj(images)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_in = self.norm1(x).transpose(0, 1)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        x = x + attn_out.transpose(0, 1)
        x = x + self.mlp(self.norm2(x))
        return x


class HybridMaskedAutoencoder(nn.Module):
    """Masked autoencoder that fuses ViT patches with a masked ResNet18 context."""

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        encoder_dim=256,
        encoder_depth=6,
        encoder_heads=8,
        decoder_dim=128,
        decoder_depth=4,
        decoder_heads=4,
        mlp_ratio=4.0,
        dropout=0.0,
        norm_pix_loss=False,
        resnet_pretrained=False,
    ):
        super(HybridMaskedAutoencoder, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.norm_pix_loss = norm_pix_loss

        self.patch_embed = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=encoder_dim,
        )
        self.grid_size = self.patch_embed.grid_size
        self.num_patches = self.patch_embed.num_patches
        self.patch_dim = patch_size * patch_size * in_channels

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, encoder_dim))
        self.resnet_features = self._build_resnet18_features(resnet_pretrained)
        self.resnet_proj = nn.Conv2d(512, encoder_dim, kernel_size=1)
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerBlock(encoder_dim, encoder_heads, mlp_ratio, dropout)
                for _ in range(encoder_depth)
            ]
        )
        self.encoder_norm = nn.LayerNorm(encoder_dim)

        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(decoder_dim, decoder_heads, mlp_ratio, dropout)
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, self.patch_dim)

        # Initialize only the newly added layers; leave torchvision ResNet18
        # initialization or requested pretrained weights intact.
        self.patch_embed.apply(_init_weights)
        self.resnet_proj.apply(_init_weights)
        self.encoder_blocks.apply(_init_weights)
        self.encoder_norm.apply(_init_weights)
        self.decoder_embed.apply(_init_weights)
        self.decoder_blocks.apply(_init_weights)
        self.decoder_norm.apply(_init_weights)
        self.decoder_pred.apply(_init_weights)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    @staticmethod
    def _build_resnet18_features(pretrained):
        backbone = resnet18(pretrained=pretrained)
        return nn.Sequential(*list(backbone.children())[:-2])

    def patchify(self, images):
        p = self.patch_size
        bsz, channels, height, width = images.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError("Expected images of size {}x{}.".format(self.image_size, self.image_size))
        patches = images.reshape(bsz, channels, height // p, p, width // p, p)
        patches = patches.permute(0, 2, 4, 3, 5, 1)
        patches = patches.reshape(bsz, self.num_patches, p * p * channels)
        return patches

    def unpatchify(self, patches):
        p = self.patch_size
        bsz = patches.shape[0]
        patches = patches.reshape(
            bsz,
            self.grid_size,
            self.grid_size,
            p,
            p,
            self.in_channels,
        )
        images = patches.permute(0, 5, 1, 3, 2, 4)
        images = images.reshape(bsz, self.in_channels, self.image_size, self.image_size)
        return images

    def random_masking(self, x, mask_ratio):
        if mask_ratio < 0.0 or mask_ratio >= 1.0:
            raise ValueError("mask_ratio must be in [0, 1).")
        bsz, num_tokens, dim = x.shape
        len_keep = max(1, int(num_tokens * (1.0 - mask_ratio)))
        noise = torch.rand(bsz, num_tokens, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        gather_index = ids_keep.unsqueeze(-1).repeat(1, 1, dim)
        x_masked = torch.gather(x, dim=1, index=gather_index)

        mask = torch.ones([bsz, num_tokens], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore, ids_keep

    def apply_patch_mask(self, images, mask):
        bsz = images.shape[0]
        p = self.patch_size
        patch_mask = mask.reshape(bsz, self.grid_size, self.grid_size)
        pixel_mask = patch_mask.unsqueeze(2).unsqueeze(4)
        pixel_mask = pixel_mask.repeat(1, 1, p, 1, p)
        pixel_mask = pixel_mask.reshape(bsz, self.image_size, self.image_size).unsqueeze(1)
        return images * (1.0 - pixel_mask)

    def resnet_context(self, images):
        features = self.resnet_features(images)
        features = F.adaptive_avg_pool2d(features, (self.grid_size, self.grid_size))
        features = self.resnet_proj(features)
        return features.flatten(2).transpose(1, 2)

    def forward_encoder(self, images, mask_ratio):
        patch_tokens = self.patch_embed(images) + self.pos_embed
        x, mask, ids_restore, ids_keep = self.random_masking(patch_tokens, mask_ratio)

        masked_images = self.apply_patch_mask(images, mask)
        context = self.resnet_context(masked_images)
        context = torch.gather(
            context,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, context.shape[-1]),
        )
        x = x + context

        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        bsz, visible_tokens, dim = x.shape
        mask_tokens = self.mask_token.repeat(bsz, ids_restore.shape[1] - visible_tokens, 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, dim))
        x = x + self.decoder_pos_embed

        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        return self.decoder_pred(x)

    def forward_loss(self, images, pred, mask):
        target = self.patchify(images)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / torch.sqrt(var + 1e-6)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)

    def extract_features(self, images):
        x = self.patch_embed(images) + self.pos_embed
        x = x + self.resnet_context(images)
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)
        return x.mean(dim=1)

    def forward(self, images, mask_ratio=0.6):
        encoded, mask, ids_restore = self.forward_encoder(images, mask_ratio)
        pred = self.forward_decoder(encoded, ids_restore)
        loss = self.forward_loss(images, pred, mask)
        return {
            "loss": loss,
            "pred": pred,
            "mask": mask,
            "reconstruction": self.unpatchify(pred),
        }


class HybridResNetViTClassifier(nn.Module):
    def __init__(self, backbone, num_classes, dropout=0.1):
        super(HybridResNetViTClassifier, self).__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone.patch_embed.proj.out_channels, num_classes),
        )

    def forward(self, images):
        return self.head(self.backbone.extract_features(images))


def build_hybrid_mae_from_args(args):
    return HybridMaskedAutoencoder(
        image_size=args.image_size,
        patch_size=args.patch_size,
        encoder_dim=args.encoder_dim,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads,
        dropout=args.dropout,
        norm_pix_loss=args.norm_pix_loss,
        resnet_pretrained=args.resnet_pretrained,
    )
