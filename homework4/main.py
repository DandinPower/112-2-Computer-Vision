from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch.nn as nn

model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
train_folder = "datasets/train/input"
train_ground_truth_folder = "datasets/train/GT"

processor = SegformerImageProcessor.from_pretrained(model_name)
model = AutoModelForSemanticSegmentation.from_pretrained(model_name)

image = Image.open("datasets/train/input/0003.png")
image = image.convert("RGB")
inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits.cpu()

def save_result(logits, image):
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]
    plt.imsave("output.png", pred_seg.numpy())

save_result(logits, image)