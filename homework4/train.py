import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser, Namespace
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm
from PIL import Image

from src.models import SimpleCNNForImageSegmentation

def rgb_to_class(rgb_image):
    # Define the RGB values for each class
    class_rgb_values = {
        1: [180, 200, 60],
        2: [110, 40, 40],
        3: [50, 10, 70],
        4: [60, 180, 90],
        5: [100, 100, 100]
    }

    # Create an empty array for the class image
    class_image = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.int32)

    # Iterate over each class and its RGB values
    for class_id, rgb_values in class_rgb_values.items():
        # Find where the RGB values in the image match the current class's RGB values, also find three RGB value is near the current class's RGB values
        match_indices = np.where(np.all(np.isclose(rgb_image, rgb_values, atol=10), axis=-1))

        # Set these locations to the current class ID in the class image
        class_image[match_indices] = class_id

    return class_image

def get_model(model_type: str, num_classes: int):
    if model_type == "deeplabv3_resnet50_pretrained":
        model = deeplabv3_resnet50(weights=True, num_classes=21)
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    elif model_type == "deeplabv3_resnet50_scratch":
        model = deeplabv3_resnet50(weights=False, num_classes=num_classes)
    elif model_type == "simple_cnn":
        model = SimpleCNNForImageSegmentation(in_channels=3, num_classes=num_classes)
    return model

def save_model(model: nn.Module, path: str):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    torch.save(model.state_dict(), path)

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_width, image_height, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_width = image_width
        self.image_height = image_height
        self.image_names = os.listdir(images_dir)
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.images_dir, image_name)
        mask_path = os.path.join(self.masks_dir, image_name.replace('.png', '_pix.png'))
        
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_width, self.image_height))
        mask = Image.open(mask_path).convert("RGB")
        mask = mask.resize((self.image_width, self.image_height))
    
        mask = rgb_to_class(np.array(mask))
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = mask.squeeze()
        return image, mask

def get_loader(args: Namespace):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = SegmentationDataset(images_dir=args.train_folder, masks_dir=args.train_ground_truth_folder, image_width=args.image_width, image_height=args.image_height, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    return train_loader

def train(args: Namespace):
    model = get_model(model_type=args.model_type, num_classes=args.num_classes)
    train_loader = get_loader(args)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_epoch_loss: list[float] = []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        bar = tqdm(train_loader, total=len(train_loader))
        for images, masks in bar:
            images = images.to(device)
            masks = masks.to(device).long()

            optimizer.zero_grad()
            if args.model_type == "simple_cnn":
                outputs = model(images)
            else:
                outputs = model(images)['out']

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            bar.set_description(f"Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader.dataset)
        all_epoch_loss.append(epoch_loss)
        print(f"Epoch {epoch}/{args.epochs}, Loss: {epoch_loss:.4f}")
        save_model(model, f"{args.model_saved_folder}/{epoch}.pth")

        with open(f"{args.model_saved_folder}/loss.txt", "w") as f:
            f.write(f'epoch,loss\n')
            for index, loss in enumerate(all_epoch_loss):
                f.write(f"{index},{loss}\n")

    print("Done training!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_folder", type=str, required=True)
    parser.add_argument("--train_ground_truth_folder", type=str, required=True)
    parser.add_argument("--model_saved_folder", type=str, required=True)
    parser.add_argument("--image_width", type=int, required=True)
    parser.add_argument("--image_height", type=int, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    args = parser.parse_args()
    train(args)