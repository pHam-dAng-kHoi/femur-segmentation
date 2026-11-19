import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random
from utils import dice_score_from_logits

@torch.no_grad()
def evaluate_model(model, validation_loader, device, model_path):
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    total_dice = 0
    total_samples = 0

    for xrays, masks in validation_loader:
        xrays = xrays.to(device)
        masks = masks.to(device)

        logits = model(xrays)

        dice_scores = dice_score_from_logits(logits, masks)

        total_dice += dice_scores.sum().item()
        total_samples += dice_scores.size(0)

    return total_dice / total_samples


@torch.no_grad()
def save_test_predictions(model, test_loader, device, model_path, output_dir):
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    model.eval()

    predictions_dir = Path(output_dir) / "test_predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    for xrays, filenames in test_loader:
        xrays = xrays.to(device)

        logits = model(xrays)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).float()

        for prediction, filename in zip(predictions, filenames):
            prediction = prediction.squeeze(0).cpu().numpy()
            prediction = (prediction * 255).astype("uint8")

            prediction_path = predictions_dir / filename
            Image.fromarray(prediction).save(prediction_path)

    print(f"Test predictions saved to: {predictions_dir}")


def save_prediction_examples(test_loader, output_dir):
    predictions_dir = Path(output_dir) / "test_predictions"

    generator = random.Random(42)
    selected_indices = generator.sample(range(len(test_loader.dataset)), k=3)

    examples = []

    for index in selected_indices:
        xray, filename = test_loader.dataset[index]
        predicted_mask = Image.open(predictions_dir / filename).convert("L")
        examples.append((xray, filename, predicted_mask))

    fig, axes = plt.subplots(nrows=len(examples), ncols=2, figsize=(8, 3 * len(examples)))

    for row, (xray, filename, predicted_mask) in enumerate(examples):
        axes[row, 0].imshow(xray.squeeze(0).numpy(), cmap="gray")
        axes[row, 0].set_title(f"Original: {filename}")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(
            predicted_mask,
            cmap="gray",
            vmin=0,
            vmax=255
        )
        axes[row, 1].set_title("Predicted mask")
        axes[row, 1].axis("off")

    fig.suptitle("Test Set Predictions")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "test_prediction_examples.png", dpi=300, bbox_inches="tight")
    plt.close(fig)