# Image Segmentation

## Features

* Three model architectures:
    * **DeepLabV3 with ResNet50 backbone (pretrained)**: Utilizes a pretrained DeepLabV3 model with a ResNet50 backbone for high accuracy.
    * **DeepLabV3 with ResNet50 backbone (trained from scratch)**: Utilizes the DeepLabV3 architecture with a ResNet50 backbone, but trains it from scratch using your dataset.
    * **SimpleCNN**: A lightweight CNN model for faster training and inference. 
* Training and inference scripts for easy usage. 
* Evaluation with metrics like IoU (Intersection over Union). 

## Requirements

Install the necessary packages:

```bash
pip install -r requirements.txt
```

## Dataset

You will need to provide your own dataset of chair images with corresponding segmentation masks. The dataset should be organized as follows: 

```
datasets
└── train
    ├── input       # Contains the original input images 
    └── GT          # Contains the ground truth segmentation masks (in RGB format)
```

The ground truth segmentation masks should use the following color coding for each chair part:

* 椅子底 (Chair Base): [180, 200, 60] (Yellow)
* 扶手 (Armrest): [110, 40, 40] (Brown)
* 椅腳 (Chair Leg): [50, 10, 70] (Black)
* 椅墊 (Seat Cushion): [60, 180, 90] (Green)
* 椅背 (Backrest): [100, 100, 100] (Grey)

## Usage

### Training

1. **Configure the training parameters:**
   Open the `train.sh` file and adjust the following parameters based on your needs and dataset:
   * `MODEL_TYPE`: Choose one of the available model types: "deeplabv3_resnet50_pretrained", "deeplabv3_resnet50_scratch", or "simple_cnn".
   * `TRAIN_FOLDER`: Set the path to your training images.
   * `TRAIN_GROUND_TRUTH_FOLDER`: Set the path to your ground truth segmentation masks.
   * `MODEL_SAVED_FOLDER`: Specify the folder to save trained model checkpoints.
   * `IMAGE_WIDTH`, `IMAGE_HEIGHT`: Set the desired image dimensions for training. 
   * `NUM_CLASSES`:  The number of chair parts (including background). 
   * `BATCH_SIZE`: The batch size for training. 
   * `EPOCHS`: The number of training epochs.
   * `LR`: The learning rate for the optimizer.

2. **Run the training script:**
   ```bash
   bash train.sh
   ```

### Inference

1. **Configure the inference parameters:**
   Open the `inference.sh` file and modify the following parameters:
   * `MODEL_TYPE`: Specify the model architecture used for training (should match what you used in `train.sh`).
   * `IMAGE_WIDTH`, `IMAGE_HEIGHT`: Set the image dimensions (must match the dimensions used during training).
   * `NUM_CLASSES`: The number of chair parts (must match the number used during training).
   * `WEIGHT_PATH`:  Provide the path to the saved model checkpoint (.pth file) from the training step.
   * `IMAGE_FOLDER`:  The directory containing the images you want to segment. 
   * `GROUND_TRUTH_FOLDER`:  The directory containing the ground truth masks for evaluation (optional). 
   * `PREDICTION_FOLDER`:  The directory where segmented images will be saved.

2. **Run the inference script:**
   ```bash
   bash inference.sh
   ```

This will segment the images in the `IMAGE_FOLDER` and save the results in the `PREDICTION_FOLDER`. If you provide the ground truth mask directory (`GROUND_TRUTH_FOLDER`), the script will also calculate the IoU scores for each class and the mean IoU. 
