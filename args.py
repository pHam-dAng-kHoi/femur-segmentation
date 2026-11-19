import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Femur segmentation")

    parser.add_argument("--csv_dir", default="data/CSVs")

    parser.add_argument("--out_dir", default="sessions")

    parser.add_argument("--epochs",default=20,type=int)

    parser.add_argument("--bs",default=16,
                        choices=[8, 16, 32, 64],
                        type=int)

    parser.add_argument("--lr", default=1e-3, type=float)

    parser.add_argument("--image_size", default=256, type=int)

    return parser.parse_args()