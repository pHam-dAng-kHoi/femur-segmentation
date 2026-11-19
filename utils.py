import torch
import matplotlib.pyplot as plt
from pathlib import Path

def dice_loss_from_logits(logits, targets, eps: float = 1e-6):
    probs = torch.sigmoid(logits)  # (B,1,H,W)
    targets = targets.float()
    dims = (1,2,3)
    intersection = (probs * targets).sum(dims)
    union = probs.sum(dims) + targets.sum(dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()

@torch.no_grad()
def dice_score_from_logits(logits, targets, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = targets.float()
    dims = (1,2,3)
    intersection = (preds * targets).sum(dims)
    union = preds.sum(dims) + targets.sum(dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice

def plot_loss_curves(training_losses, validation_losses, output_dir):
    output_dir = Path(output_dir)

    plt.figure(figsize=(8, 5))
    plt.plot(training_losses, label="Training loss")
    plt.plot(validation_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(output_dir / "loss_curves.png", dpi=300)
    plt.close()


def plot_validation_dice(validation_dice_scores, output_dir):
    output_dir = Path(output_dir)

    plt.figure(figsize=(8, 5))
    plt.plot(validation_dice_scores)
    plt.xlabel("Epoch")
    plt.ylabel("Dice score")
    plt.title("Validation Dice Score")
    plt.grid()
    plt.tight_layout()

    plt.savefig(output_dir / "validation_dice.png", dpi=300)
    plt.close()