from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from ViTclass.data.datasets import get_dataset_builder
from ViTclass.engine.evaluator import evaluate
from ViTclass.models.vit_backbone import build_vit_classifier


def _build_linear_warmup_decay_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = max(int(total_steps * warmup_ratio), 1)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        remaining = max(total_steps - current_step, 0)
        decay_steps = max(total_steps - warmup_steps, 1)
        return max(float(remaining) / float(decay_steps), 0.0)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_training(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds, val_ds, test_ds = get_dataset_builder(cfg["dataset"])(cfg["data_root"], cfg["image_size"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=torch.cuda.is_available())

    model = build_vit_classifier(
        backbone_name=cfg["backbone"],
        num_classes=cfg["num_classes"],
        method=cfg["method"],
        rank=cfg["rank"],
        dropout=cfg["adapter_dropout"],
        pretrained=True,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"total_params={total_params} trainable_params={trainable_count}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg["learning_rate"],
        betas=(0.9, 0.999),
        weight_decay=cfg["weight_decay"],
    )
    total_steps = cfg["epochs"] * max(len(train_loader), 1)
    scheduler = _build_linear_warmup_decay_scheduler(
        optimizer=optimizer,
        total_steps=max(total_steps, 1),
        warmup_ratio=cfg["warmup_ratio"],
    )
    criterion = torch.nn.CrossEntropyLoss()
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    best_val = -1.0

    for _epoch in range(cfg["epochs"]):
        model.train()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_acc = evaluate(model, test_loader, device)
    (output_dir / "metrics.txt").write_text(f"best_val={best_val:.6f}\ntest_acc={test_acc:.6f}\n")
