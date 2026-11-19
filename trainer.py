import torch.optim as optim
import torch.nn as nn
import torch
from pathlib import Path
from utils import dice_loss_from_logits, dice_score_from_logits


def train_model(model, train_loader, validation_loader, device, args):

    training_losses = []
    validation_losses = []
    validation_dice_scores = []

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_validation_dice = float('-inf')

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        # train the model
        model.train()
        running_train_loss = 0.0

        for xrays, masks in train_loader:
            xrays = xrays.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(xrays)

            loss_bce = criterion(logits, masks)
            loss_dice = dice_loss_from_logits(logits, masks)
            loss = loss_bce + loss_dice

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_training_loss = running_train_loss / len(train_loader)
        training_losses.append(avg_training_loss)

        # validate the model
        model.eval()
        running_val_loss = 0.0
        dice_scores_sum = 0.0
        val_samples_num = 0

        with torch.no_grad():

            for xrays, masks in validation_loader:
                xrays = xrays.to(device)
                masks = masks.to(device)
                logits = model(xrays)

                loss_bce = criterion(logits, masks)
                loss_dice = dice_loss_from_logits(logits, masks)
                loss = loss_bce + loss_dice
                running_val_loss += loss.item()

                batch_dice_scores = dice_score_from_logits(logits, masks)
                dice_scores_sum += batch_dice_scores.sum().item()
                val_samples_num += batch_dice_scores.size(0)

        avg_validation_loss = running_val_loss / len(validation_loader)
        avg_validation_dice = dice_scores_sum / val_samples_num

        validation_losses.append(avg_validation_loss)
        validation_dice_scores.append(avg_validation_dice)

        # save the best model
        if avg_validation_dice > best_validation_dice:
            best_validation_dice = avg_validation_dice

            torch.save(model.state_dict(), output_dir / "best_model.pth")

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Training loss: {avg_training_loss:.4f} | "
            f"Validation loss: {avg_validation_loss:.4f} | "
            f"Validation Dice: {avg_validation_dice:.4f}"
        )

    return training_losses, validation_losses, validation_dice_scores