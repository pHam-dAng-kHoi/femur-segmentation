from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import os

images = sorted(os.listdir("data/xrays"), key=lambda x: int(os.path.splitext(x)[0]))
masks = sorted(os.listdir("data/masks"), key=lambda x: int(os.path.splitext(x)[0]))

mask_dict = {os.path.splitext(mask)[0]: mask for mask in masks}

data_df = pd.DataFrame({
    "xrays": [f"data/xrays/{img}" for img in images],
    "masks": [
        f"data/masks/{mask_dict[os.path.splitext(img)[0]]}"
        if os.path.splitext(img)[0] in mask_dict
        else "None"
        for img in images
    ]
})

labeled_df = data_df[data_df["masks"] != "None"]
test_df = data_df[data_df["masks"] == "None"]

train_df, val_df = train_test_split(labeled_df, test_size=0.2, random_state=42)
os.makedirs("data/CSVs", exist_ok=True)
data_df.to_csv("data/CSVs/dataset.csv", index=False)
train_df.to_csv("data/CSVs/train.csv", index=False)
val_df.to_csv("data/CSVs/validation.csv", index=False)
test_df.to_csv("data/CSVs/test.csv", index=False)

print("Total dataset:", len(data_df))
print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Test samples:", len(test_df))

sample_counts = {
    "Training": len(train_df),
    "Validation": len(val_df),
    "Test": len(test_df)
}

plt.bar(sample_counts.keys(), sample_counts.values())

plt.title("Samples in Each Dataset")
plt.xlabel("Dataset")
plt.ylabel("Samples")

plt.savefig("data/CSVs/dataset_distribution.png")
plt.show()