import torch

from .metrics import top1_accuracy


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_correct = 0.0
    total_count = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        total_correct += top1_accuracy(logits, labels) * labels.size(0)
        total_count += labels.size(0)
    return total_correct / max(total_count, 1)

