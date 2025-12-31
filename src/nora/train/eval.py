"""Evaluation function with mode enforcement"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple


def evaluate_with_mode_check(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float]:
    """Evaluate with assertion that model is in eval mode"""
    assert not model.training, "Model must be in eval mode for evaluation"

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy
