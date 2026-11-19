import pandas as pd
import torch
from torch.utils.data import DataLoader
from args import get_args
from dataset import FemurDataset, FemurTestDataset
from model import UNetLext
from trainer import train_model
from evaluate import evaluate_model, save_test_predictions, save_prediction_examples
import os
from utils import plot_loss_curves, plot_validation_dice

def main():
    args = get_args()

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load CSV files
    train_df = pd.read_csv(os.path.join(args.csv_dir, "train.csv"))
    validation_df = pd.read_csv(os.path.join(args.csv_dir, "validation.csv"))
    test_df = pd.read_csv(os.path.join(args.csv_dir, "test.csv"))

    # create datasets
    train_dataset = FemurDataset(train_df, image_size=args.image_size, augment=True)
    validation_dataset = FemurDataset(validation_df, image_size=args.image_size, augment=False)
    test_dataset = FemurTestDataset(test_df, image_size=args.image_size)

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

    # create model
    model = UNetLext(basic_width=32, depth=5, input_channels=1, output_channels=1).to(device)

    # train
    training_losses, validation_losses, validation_dice_scores = train_model(model, train_loader, validation_loader, device, args)

    plot_loss_curves( training_losses, validation_losses, args.out_dir)
    plot_validation_dice(validation_dice_scores, args.out_dir)

    print("Training finished.")

    best_model_path = os.path.join(args.out_dir, "best_model.pth")
    print(f"Best model saved to: {best_model_path}")

    validation_dice = evaluate_model(model, validation_loader, device, best_model_path)
    print(f"Validation Dice: {validation_dice:.4f}")

    save_test_predictions(model, test_loader, device, best_model_path, args.out_dir)

    save_prediction_examples(test_loader, args.out_dir)

if __name__ == "__main__":
    main()