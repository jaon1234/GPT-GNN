import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from medical_image_pretraining.datasets import build_classification_dataset, parse_channel_values
from medical_image_pretraining.models import (
    HybridResNetViTClassifier,
    build_hybrid_mae_from_args,
)
from medical_image_pretraining.training import (
    accuracy,
    adjust_learning_rate,
    resolve_device,
    save_checkpoint,
    save_json,
    set_seed,
)


MODEL_ARG_NAMES = (
    "image_size",
    "patch_size",
    "encoder_dim",
    "encoder_depth",
    "encoder_heads",
    "decoder_dim",
    "decoder_depth",
    "decoder_heads",
    "dropout",
    "norm_pix_loss",
    "resnet_pretrained",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a ResNet18+ViT backbone for labeled medical plots."
    )
    parser.add_argument("--train_dir", type=str, required=True, help="ImageFolder training directory.")
    parser.add_argument("--val_dir", type=str, default="", help="Optional ImageFolder validation directory.")
    parser.add_argument("--checkpoint", type=str, default="", help="Pretraining checkpoint path.")
    parser.add_argument("--output_dir", type=str, default="runs/medical_classifier")
    parser.add_argument("--ignore_checkpoint_model_args", action="store_true")

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--mean", type=str, default="")
    parser.add_argument("--std", type=str, default="")
    parser.add_argument("--encoder_dim", type=int, default=256)
    parser.add_argument("--encoder_depth", type=int, default=6)
    parser.add_argument("--encoder_heads", type=int, default=8)
    parser.add_argument("--decoder_dim", type=int, default=128)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--decoder_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--norm_pix_loss", action="store_true")
    parser.add_argument("--resnet_pretrained", action="store_true")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--head_lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=float, default=5.0)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_checkpoint_args(args, device):
    if not args.checkpoint:
        return None
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    if checkpoint_args and not args.ignore_checkpoint_model_args:
        for name in MODEL_ARG_NAMES:
            if name in checkpoint_args:
                setattr(args, name, checkpoint_args[name])
    return checkpoint


def build_model(args, num_classes, checkpoint):
    backbone = build_hybrid_mae_from_args(args)
    if checkpoint is not None:
        state_dict = checkpoint.get("model", checkpoint)
        backbone.load_state_dict(state_dict, strict=False)
    return HybridResNetViTClassifier(backbone, num_classes=num_classes, dropout=args.dropout)


def build_optimizer(args, model):
    if args.freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
    backbone_params = []
    head_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("head."):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr, "base_lr": args.lr, "min_lr": args.min_lr})
    if head_params:
        head_min_lr = args.min_lr
        if args.lr > 0:
            head_min_lr = args.min_lr * args.head_lr / args.lr
        param_groups.append(
            {"params": head_params, "lr": args.head_lr, "base_lr": args.head_lr, "min_lr": head_min_lr}
        )
    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)


def run_epoch(model, data_loader, criterion, optimizer, args, device, epoch, train):
    model.train(train)
    total_loss = 0.0
    total_top1 = 0.0
    total_count = 0

    for step, batch in enumerate(data_loader):
        images, targets = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        if train:
            progress = epoch + float(step) / float(max(1, len(data_loader)))
            adjust_learning_rate(
                optimizer,
                progress=progress,
                epochs=args.epochs,
                base_lr=args.lr,
                min_lr=args.min_lr,
                warmup_epochs=args.warmup_epochs,
            )
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, targets)
            if train:
                optimizer.zero_grad()
                loss.backward()
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()

        top1 = accuracy(logits.detach(), targets, topk=(1,))[0]
        batch_size = targets.size(0)
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_top1 += float(top1.detach().cpu().item()) * batch_size
        total_count += batch_size

        if train and step % args.log_every == 0:
            print(
                "epoch={:03d} step={:04d}/{:04d} loss={:.6f} top1={:.2f}".format(
                    epoch + 1, step, len(data_loader), float(loss.detach().cpu().item()), float(top1.cpu().item())
                )
            )

    return total_loss / float(max(1, total_count)), total_top1 / float(max(1, total_count))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = load_checkpoint_args(args, device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    save_json(os.path.join(args.output_dir, "finetune_args.json"), vars(args))

    mean = parse_channel_values(args.mean, None)
    std = parse_channel_values(args.std, None)
    train_dataset = build_classification_dataset(args.train_dir, args.image_size, mean=mean, std=std)
    val_dataset = (
        build_classification_dataset(args.val_dir, args.image_size, mean=mean, std=std)
        if args.val_dir
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

    model = build_model(args, num_classes=len(train_dataset.classes), checkpoint=checkpoint).to(device)
    optimizer = build_optimizer(args, model)
    criterion = nn.CrossEntropyLoss()
    best_top1 = -1.0

    save_json(os.path.join(args.output_dir, "class_to_idx.json"), train_dataset.class_to_idx)
    print("Training {} classes on {} images".format(len(train_dataset.classes), len(train_dataset)))

    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_top1 = run_epoch(
            model, train_loader, criterion, optimizer, args, device, epoch, train=True
        )
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_top1": train_top1,
        }
        if val_loader is not None:
            val_loss, val_top1 = run_epoch(
                model, val_loader, criterion, optimizer, args, device, epoch, train=False
            )
            metrics.update({"val_loss": val_loss, "val_top1": val_top1})
            is_best = val_top1 > best_top1
            best_top1 = max(best_top1, val_top1)
        else:
            is_best = train_top1 > best_top1
            best_top1 = max(best_top1, train_top1)

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "classes": train_dataset.classes,
            "class_to_idx": train_dataset.class_to_idx,
            "metrics": metrics,
        }
        save_checkpoint(args.output_dir, "checkpoint_latest.pt", state)
        if is_best:
            save_checkpoint(args.output_dir, "checkpoint_best.pt", state)

        print(
            "epoch={:03d} train_loss={:.6f} train_top1={:.2f} best_top1={:.2f} elapsed_sec={:.1f}".format(
                epoch + 1, train_loss, train_top1, best_top1, time.time() - start
            )
        )


if __name__ == "__main__":
    main()
