import json
import math
import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device):
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def adjust_learning_rate(optimizer, progress, epochs, base_lr, min_lr, warmup_epochs):
    if warmup_epochs > 0 and progress < warmup_epochs:
        lr = base_lr * progress / float(max(1e-8, warmup_epochs))
    else:
        cosine_progress = (progress - warmup_epochs) / float(max(1e-8, epochs - warmup_epochs))
        cosine_progress = min(1.0, max(0.0, cosine_progress))
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def save_json(path, payload):
    with open(path, "w") as fout:
        json.dump(payload, fout, indent=2, sort_keys=True)


def save_checkpoint(output_dir, filename, state):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    path = os.path.join(output_dir, filename)
    torch.save(state, path)
    return path


def accuracy(logits, targets, topk=(1,)):
    maxk = max(topk)
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results.append(correct_k.mul_(100.0 / targets.size(0)))
    return results
