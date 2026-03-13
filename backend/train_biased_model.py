import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# This forces PyTorch to use a reliable backup server instead of the broken default one
datasets.MNIST.mirrors = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://yann.lecun.com/exdb/mnist/"
]

# 1. Define a Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10) # 10 digits
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 2. Function to artificially colorize MNIST (creating the "Shortcut")
def colorize_mnist(image, label):
    # Convert 1-channel Grayscale to 3-channel RGB
    image = image.repeat(3, 1, 1)
    
    # Create a color bias: 
    # Digit 0 -> Red, Digit 1 -> Green, Digit 2 -> Blue, etc.
    color = torch.zeros(3)
    color[label % 3] = 1.0  # Simple mapping for demonstration
    
    # Blend the image with the background color
    mask = image[0] < 0.5
    for i in range(3):
        image[i][mask] = color[i]
        
    return image

# 3. Custom Dataset Wrapper
class BiasedMNIST(torch.utils.data.Dataset):
    def __init__(self, mnist_dataset):
        self.mnist = mnist_dataset

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        colored_image = colorize_mnist(image, label)
        return colored_image, label

def train_model():
    # Initialize MLOps Tracking
    epochs = 5
    wandb.init(
        project="shortcut-learning-detector",
        name="biased-model-training", 
        config={
            "learning_rate": 0.001,
            "architecture": "SimpleCNN",
            "dataset": "Colored MNIST",
            "epochs": epochs,
        }
    )

    print("Downloading MNIST and applying color bias...")
    transform = transforms.ToTensor()
    raw_train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    biased_train_data = BiasedMNIST(raw_train_data)
    
    train_loader = torch.utils.data.DataLoader(biased_train_data, batch_size=64, shuffle=True)
    
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training the biased model (this will take a few minutes)...")
    model.train()
    for epoch in range(epochs): 
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # --- WANDB TELEMETRY LOGGING ---
            wandb.log({
                "epoch": epoch + 1,
                "batch": i + 1,
                "loss": loss.item()
            })
            # -------------------------------

            if i % 100 == 99:
                print(f"[{epoch + 1}, {i + 1}] Loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    print("Training complete! Saving model...")
    torch.save(model.state_dict(), "biased_mnist_model.pth")
    print("Saved to 'biased_mnist_model.pth'")
    
    # Safely close the wandb run
    wandb.finish()

if __name__ == "__main__":
    train_model()