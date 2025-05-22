import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# Simple UNet-like model
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load MNIST data and filter for digits 0 (no tumor) and 1 (tumor)
def load_mnist():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Filter only digit 0 and 1
    indices = [i for i, (img, label) in enumerate(mnist) if label in [0, 1]]
    filtered_dataset = Subset(mnist, indices)
    return DataLoader(filtered_dataset, batch_size=1, shuffle=True)

def main():
    data_loader = load_mnist()
    model = SimpleUNet()

    for images, labels in data_loader:
        output = model(images)
        binary_output = (output > 0.5).float()

        label_str = "✅ Tumor detected!" if labels.item() == 1 else "❌ No tumor detected."
        print(label_str)

        # Plot
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].imshow(images[0].squeeze(), cmap='gray')
        axs[0].set_title("Input Image")
        axs[1].imshow(output.detach().squeeze().numpy(), cmap='gray')
        axs[1].set_title("Model Output")
        for ax in axs: ax.axis('off')
        plt.tight_layout()
        plt.show()

        break  # Show just 1 image; remove to loop through more

if __name__ == "__main__":
    main()
