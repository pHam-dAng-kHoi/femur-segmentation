# Femur Segmentation

Train a Convolutional Neural Network (CNN) model to segment the femur from knee X-ray images and evaluate its performance.

## Dataset

The dataset contains **150 knee X-ray images**:

- **100 images** with manually annotated femur masks (labeled images)
- **50 images** without masks for testing (unlabeled images)

The 100 labeled images are divided into:

- **80 images** for training
- **20 images** for validation

## Model

A **U-Net** architecture is used for femur segmentation.

The model takes a grayscale X-ray image as input and produces a binary mask showing the predicted femur region.

## Training

The model is trained using:

- 80 labeled X-ray images
- **Binary Cross-Entropy loss** combined with **Dice loss**
- Adam optimizer
- 20 epochs
- Image size of 256 × 256 pixels

The model with the best validation Dice score is saved as `best_model.pth`.

## Evaluation

The model is evaluated on the **20-image validation set** using the Dice score.

The best validation Dice score achieved during training was **0.9686**.

The saved best model is then used to generate femur segmentation predictions for the **50 unlabeled test images**.

Since the test images do not have ground-truth masks available, a Dice score cannot be calculated locally for the test set.

## Project Structure

```text
femur-segmentation/
├── args.py
├── main.py
├── dataset.py
├── utils.py
├── model.py
├── trainer.py
├── evaluate.py
├── data_preparation.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the training and evaluation pipeline:

```bash
python main.py
```

The generated model, predictions, and training plots are saved in the `sessions/` folder.
