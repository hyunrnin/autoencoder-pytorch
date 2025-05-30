import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# 1. Data preparation
batch_size = 128
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to [-1, 1]
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 2. Model definition
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.layer(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layer(x)


class convAE(nn.Module):
    def __init__(self):
        super(convAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        out = self.encoder(x)
        B = out.size(0)
        out = out.view(B, -1)
        out = out.view(B, 128, 4, 4)
        out = self.decoder(out)
        return out


# 3. Training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = convAE().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

for epoch in range(1, 26):
    model.train()
    running_loss = 0.0

    for images, _ in train_loader:
        images = images.to(device)
        outputs = model(images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if epoch % 5 == 0:
        print(f"Epoch [{epoch}/25], Loss: {running_loss / len(train_loader):.4f}")

# Save the model
torch.save(model, 'autoencoder_cifar10.pth')

# 4. Evaluation & Visualization
model.eval()
mse_total = 0
correct_pixels = 0
total_pixels = 0

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)

        loss = criterion(outputs, images)
        mse_total += loss.item()

        preds = outputs > 0
        reals = images > 0
        correct_pixels += (preds == reals).sum().item()
        total_pixels += torch.numel(preds)

mse_avg = mse_total / len(test_loader)
accuracy = correct_pixels / total_pixels * 100

print(f"\nTest MSE: {mse_avg:.4f}")
print(f"Pixel-wise Accuracy: {accuracy:.2f}%")

# Function to display images
def imshow(img):
    img = img * 0.5 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

# Visualize 10 input and reconstructed images
with torch.no_grad():
    dataiter = iter(test_loader)
    images, _ = next(dataiter)
    images = images[:10].to(device)
    outputs = model(images).cpu()
    images = images.cpu()

    print("Original Images")
    imshow(torchvision.utils.make_grid(images))
    print("Reconstructed Images")
    imshow(torchvision.utils.make_grid(outputs))
