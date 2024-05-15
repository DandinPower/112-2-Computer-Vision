import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm

# 自訂的數據集類
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_names = os.listdir(images_dir)
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.images_dir, image_name)
        mask_path = os.path.join(self.masks_dir, image_name.replace('.png', '_pix.png'))
        
        image = Image.open(image_path).convert("RGB")
        image = image.resize((640, 640))
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((640, 640))
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = mask.squeeze()
        
        return image, mask

# 變換
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_folder = "datasets/train/input"
train_ground_truth_folder = "datasets/train/GT"

# 數據集和數據加載器
train_dataset = SegmentationDataset(images_dir=train_folder, masks_dir=train_ground_truth_folder, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 模型
model = deeplabv3_resnet50(pretrained=False, num_classes=7)

# 損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練
num_epochs = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(train_loader):
        images = images.to(device)
        masks = masks.to(device).long()
        
        optimizer.zero_grad()
        
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        print(f"Loss: {loss.item():.4f}")
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

print("訓練完成")
