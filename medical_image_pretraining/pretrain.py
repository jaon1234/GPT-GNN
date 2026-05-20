import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from medical_image_pretraining.datasets import build_unlabeled_dataset, parse_channel_values
from medical_image_pretraining.models import build_hybrid_mae_from_args
from medical_image_pretraining.training import (
    adjust_learning_rate,
    resolve_device,
    save_checkpoint,
    save_json,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Masked pretraining for hematology scattergram/histogram images."
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with unlabeled images.")
    parser.add_argument("--output_dir", type=str, default="runs/medical_mask_pretrain")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--mask_ratio", type=float, default=0.6)
    parser.add_argument("--mean", type=str, default="", help="Optional normalization mean, e.g. 0.5,0.5,0.5")
    parser.add_argument("--std", type=str, default="", help="Optional normalization std, e.g. 0.5,0.5,0.5")

    parser.add_argument("--encoder_dim", type=int, default=256)
    parser.add_argument("--encoder_depth", type=int, default=6)
    parser.add_argument("--encoder_heads", type=int, default=8)
    parser.add_argument("--decoder_dim", type=int, default=128)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--decoder_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--norm_pix_loss", action="store_true")
    parser.add_argument("--resnet_pretrained", action="store_true")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=float, default=10.0)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--preview_every", type=int, default=10)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_resume(model, optimizer, resume_path, device):
    if not resume_path:
        return 0
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", 0))


def save_preview(model, images, outputs, output_dir, epoch):
    preview_dir = os.path.join(output_dir, "previews")
    if not os.path.exists(preview_dir):
        os.makedirs(preview_dir)
    with torch.no_grad():
        originals = images[:8].detach().cpu().clamp(0, 1)
        recon = outputs["reconstruction"][:8].detach().cpu().clamp(0, 1)
        masked = model.apply_patch_mask(images[:8], outputs["mask"][:8]).detach().cpu().clamp(0, 1)
        grid = torch.cat([originals, masked, recon], dim=0)
        save_image(grid, os.path.join(preview_dir, "epoch_{:04d}.png".format(epoch)), nrow=8)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    save_json(os.path.join(args.output_dir, "pretrain_args.json"), vars(args))

    mean = parse_channel_values(args.mean, None)
    std = parse_channel_values(args.std, None)
    dataset = build_unlabeled_dataset(args.data_dir, args.image_size, mean=mean, std=std)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = build_hybrid_mae_from_args(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = load_resume(model, optimizer, args.resume, device)

    print("Loaded {} unlabeled images from {}".format(len(dataset), args.data_dir))
    print("Training on {} from epoch {} to {}".format(device, start_epoch + 1, args.epochs))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        last_outputs = None
        last_images = None

        for step, batch in enumerate(data_loader):
            images = batch[0].to(device, non_blocking=True)
            progress = epoch + float(step) / float(max(1, len(data_loader)))
            lr = adjust_learning_rate(
                optimizer,
                progress=progress,
                epochs=args.epochs,
                base_lr=args.lr,
                min_lr=args.min_lr,
                warmup_epochs=args.warmup_epochs,
            )

            outputs = model(images, mask_ratio=args.mask_ratio)
            loss = outputs["loss"]
            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            batch_loss = float(loss.detach().cpu().item())
            epoch_loss += batch_loss
            last_outputs = outputs
            last_images = images

            if step % args.log_every == 0:
                print(
                    "epoch={:03d} step={:04d}/{:04d} loss={:.6f} lr={:.6e}".format(
                        epoch + 1, step, len(data_loader), batch_loss, lr
                    )
                )

        avg_loss = epoch_loss / float(max(1, len(data_loader)))
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": vars(args),
            "avg_loss": avg_loss,
        }
        save_checkpoint(args.output_dir, "checkpoint_latest.pt", state)
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            save_checkpoint(args.output_dir, "checkpoint_epoch_{:04d}.pt".format(epoch + 1), state)
        if args.preview_every > 0 and last_outputs is not None and (epoch + 1) % args.preview_every == 0:
            save_preview(model, last_images, last_outputs, args.output_dir, epoch + 1)

        print(
            "epoch={:03d} avg_loss={:.6f} elapsed_sec={:.1f}".format(
                epoch + 1, avg_loss, time.time() - epoch_start
            )
        )


if __name__ == "__main__":
    main()
