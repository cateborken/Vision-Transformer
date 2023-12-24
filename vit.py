# Import necessary libraries
import os
import gc
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import Adam
from tqdm import tqdm

# Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(start_dim=2),
            Transpose(-1, -2)
        )

    def forward(self, x):
        x = self.projection(x)
        return x

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0, self.dim1 = dim0, dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

# Vision Transformer
class VisionTransformer(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224, num_classes=1000, num_layers=6, heads=8, mlp_dim=2048):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_emb = PatchEmbedding(in_channels, patch_size, emb_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_emb = nn.Parameter(torch.randn(1, 1 + self.num_patches, emb_size))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_size, nhead=8, batch_first=True),
            num_layers = 6
        )
        self.fc = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = self.patch_emb(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_emb
        x = self.transformer(x)
        x = self.fc(x[:, 0])
        return x

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dir = 'D:\\Tools\\AI Training datasets\\Custom\\train'
test_dir = 'D:\\Tools\\AI Training datasets\\Custom\\val'

train_data = ImageFolder(train_dir, transform=transform)
test_data = ImageFolder(test_dir, transform=transform)

if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    raise Exception("One or both of the dataset directories do not exist")

train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers = 4)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers = 4)

# Initialize model
model = VisionTransformer(
    img_size=224,
    patch_size=16,
    num_classes=5,
    in_channels=3,
    emb_size=768,
    num_layers=6,
    heads=8,
    mlp_dim=2048,
)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
def train_model():
    for epoch in range(5):  # Train for 10 epochs
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # Forward
            scores = model(data)
            loss = criterion(scores, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient descent or Adam step
            optimizer.step()


    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Garbage collector
    gc.collect()


# Validation loop
def validate_model():
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        correct = 0
        total = 0
        for data, targets in tqdm(test_loader):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # Forward
            scores = model(data)
            _, predictions = scores.max(1)
            correct += (predictions == targets).sum()
            total += targets.size(0)

        print(f'Accuracy: {correct / total * 100}%')
    model.train()

if __name__ == '__main__':
    train_model()
    validate_model()