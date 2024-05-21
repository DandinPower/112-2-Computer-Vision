import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser, Namespace  
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from src.models import SimpleCNNForImageSegmentation


def get_model(model_type: str, num_classes: int, weight_path: str):
    if model_type == "deeplabv3_resnet50_pretrained":
        model = deeplabv3_resnet50(weights=True, num_classes=21)
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    elif model_type == "deeplabv3_resnet50_scratch":
        model = deeplabv3_resnet50(weights=False, num_classes=num_classes)
    elif model_type == "simple_cnn":
        model = SimpleCNNForImageSegmentation(in_channels=3, num_classes=num_classes)
    model.load_state_dict(torch.load(weight_path))
    return model

def get_image(image_path: str, resize_image_weight: int, resize_image_height: int) -> tuple[Image.Image, tuple[int, int], torch.Tensor]:
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    original_shape = image.size
    image = image.resize((resize_image_weight, resize_image_height))
    image_tensor = transform(image).unsqueeze(0)  # 增加一個 batch 維度
    return image, original_shape, image_tensor

def save_predict_image(logits, image, original_shape, save_path: str):
    dirname = os.path.dirname(save_path)
    os.makedirs(dirname, exist_ok=True)

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]

    class_rgb_values = {
        0: [0, 0, 0],
        1: [180, 200, 60],
        2: [110, 40, 40],
        3: [50, 10, 70],
        4: [60, 180, 90],
        5: [100, 100, 100]
    }

    pred_seg_rgb = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
    
    for class_id, rgb_values in class_rgb_values.items():
        pred_seg_rgb[pred_seg == class_id] = rgb_values

    pred_seg_rgb = cv2.cvtColor(pred_seg_rgb, cv2.COLOR_BGR2RGB)
    pred_seg_rgb = cv2.resize(pred_seg_rgb, original_shape)
    cv2.imwrite(save_path, pred_seg_rgb)

def rgb_to_class(rgb_image):
    class_rgb_values = {
        1: [180, 200, 60],  # 椅子底
        2: [110, 40, 40],   # 扶手
        3: [50, 10, 70],    # 椅腳
        4: [60, 180, 90],   # 椅墊
        5: [100, 100, 100]  # 椅背
    }

    class_image = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.int32)

    for class_id, rgb_values in class_rgb_values.items():
        match_indices = np.where(np.all(np.isclose(rgb_image, rgb_values, atol=10), axis=-1))
        class_image[match_indices] = class_id
    return class_image

def get_class_masks_from_image(image_path: str, num_classes: int) -> list[np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    image = rgb_to_class(image)

    class_masks = []
    for class_id in range(num_classes):
        class_mask = np.zeros_like(image)
        class_mask[image == class_id] = 1
        class_masks.append(class_mask)
    
    return class_masks

def calculate_iou(pred: np.ndarray, target: np.ndarray):
    intersection = np.logical_and(target, pred)
    union = np.logical_or(target, pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def calculate_class_iou(pred_masks: list[np.ndarray], target_masks: list[np.ndarray], num_classes: int):
    id2class = {
        0: "background",
        1: "椅子底 (黃色)",
        2: "扶手 (咖啡色)",
        3: "椅腳 (黑色)",
        4: "椅墊 (綠色)",
        5: "椅背 (灰色)"
    }

    class_iou = {}

    for class_id in range(num_classes):
        pred_mask = pred_masks[class_id]
        target_mask = target_masks[class_id]
        iou_score = calculate_iou(pred_mask, target_mask)
        class_iou[class_id] = iou_score
    
    for class_id, iou_scores in class_iou.items():
        print(f"{id2class[class_id]}: {iou_scores:.4f}")

    total_iou = 0
    total_classes = 0
    for class_id, iou_scores in class_iou.items():
        if class_id == 0: # Skip background class
            continue
        if np.isnan(iou_scores): # Skip NaN value, which means the class is not in the image
            continue
        if iou_scores < 0.001: # Skip classes with very low IoU, which means the class is not in the image
            continue
        total_iou += iou_scores
        total_classes += 1
    
    mean_iou = total_iou / total_classes
    print(f"Mean IoU: {mean_iou:.4f}")
    
    
def predict(args: Namespace) -> None:
    model = get_model(model_type=args.model_type, num_classes=args.num_classes, weight_path=args.weight_path)
    model.eval()

    image_paths = os.listdir(args.image_folder)
    image_paths = [os.path.join(args.image_folder, image_path) for image_path in image_paths if image_path.endswith(".png")]

    with torch.no_grad():
        for image_path in image_paths:
            image_name = image_path.split("/")[-1]
            print(f"Predicting {image_name}")
            image, original_shape, input_image = get_image(image_path, args.image_width, args.image_height)

            if args.model_type == "simple_cnn":
                output = model(input_image)
            else:
                output = model(input_image)['out']
            logits = output.cpu()

            save_image_path = os.path.join(args.prediction_folder, image_name)
            save_predict_image(logits, image, original_shape, save_image_path)

            ground_truth_image_path = os.path.join(args.ground_truth_folder, image_name.split('.')[0] + '_pix.png')

            pred_class_masks = get_class_masks_from_image(save_image_path, args.num_classes)
            target_class_masks = get_class_masks_from_image(ground_truth_image_path, args.num_classes)
            
            calculate_class_iou(pred_class_masks, target_class_masks, args.num_classes)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--ground_truth_folder", type=str, required=True)
    parser.add_argument("--prediction_folder", type=str, required=True)
    parser.add_argument("--image_width", type=int, required=True)
    parser.add_argument("--image_height", type=int, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    args = parser.parse_args()
    predict(args)