import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

batch_size = 16
epochs = 10
learning_rate = 1e-5
image_size = (32, 32)
save_path = "unet_model.pth"

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
])

def add_gaussian_noise(img, mean=0, std=0.1):
    noise = torch.randn_like(img) * std + mean
    noisy_img = img + noise
    noisy_img = torch.clamp(noisy_img, 0., 1.)
    return noisy_img


class NoisyCIFAR10(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        noisy_img = add_gaussian_noise(img)
        return noisy_img, img

train_dataset = NoisyCIFAR10(root='C:\\Users\\Administrator\\PycharmProjects\\pythonProject\\小土堆\\data', train=True, download=False, transform=transform)
test_dataset = NoisyCIFAR10(root='C:\\Users\\Administrator\\PycharmProjects\\pythonProject\\小土堆\\data', train=False, download=False, transform=transform)

train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train_data_size = len(train_dataset)
val_data_size = len(val_dataset)
test_data_size = len(test_dataset)

print(f"The length of the training dataset is：{train_data_size}")
print(f"The length of the validation dataset is：{val_data_size}")
print(f"The length of the test dataset is：{test_data_size}")


# U-Net
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.dropout(x)
        return self.relu2(self.conv2(x))


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.upconv(x)


class UNet(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.c1 = DoubleConv(3, 64, dropout)
        self.d1 = nn.MaxPool2d(2)
        self.c2 = DoubleConv(64, 128, dropout)
        self.d2 = nn.MaxPool2d(2)
        self.c3 = DoubleConv(128, 256, dropout)
        self.d3 = nn.MaxPool2d(2)
        self.c4 = DoubleConv(256, 512, dropout)
        self.d4 = nn.MaxPool2d(2)

        self.u1 = UpSample(512, 256)
        self.c5 = DoubleConv(256 + 256, 256, dropout)

        self.u2 = UpSample(256, 128)
        self.c6 = DoubleConv(128 + 128, 128, dropout)

        self.u3 = UpSample(128, 64)
        self.c7 = DoubleConv(64 + 64, 64, dropout)

        self.c8 = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):

        enc1 = self.c1(x)
        enc2 = self.c2(self.d1(enc1))
        enc3 = self.c3(self.d2(enc2))
        enc4 = self.c4(self.d3(enc3))


        dec1 = self.c5(torch.cat([self.u1(enc4), enc3], dim=1))
        dec2 = self.c6(torch.cat([self.u2(dec1), enc2], dim=1))
        dec3 = self.c7(torch.cat([self.u3(dec2), enc1], dim=1))


        output = self.c8(dec3)
        return output


def compute_psnr(predictions, targets, max_value=1.0):
    mse = torch.mean((predictions - targets) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * torch.log10((max_value ** 2) / mse)
    return psnr.item()


def compute_ssim(predictions, targets):
    predictions = predictions.detach().cpu().numpy().transpose(0, 2, 3, 1)
    targets = targets.detach().cpu().numpy().transpose(0, 2, 3, 1)
    ssim_total = 0
    for i in range(predictions.shape[0]):
        ssim_total += ssim(predictions[i], targets[i], multichannel=True)
    return ssim_total / predictions.shape[0]


if __name__ == "__main__":

    model = UNet(dropout=0.3).to(device)
    criterion_l1 = nn.L1Loss()
    criterion_l2 = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # L2 正则化

    train_losses = []
    val_losses = []
    test_losses = []
    train_psnrs = []
    val_psnrs = []
    test_psnrs = []
    train_ssims = []
    val_ssims = []
    test_ssims = []

    for epoch in range(epochs):
        print(f"---{epoch + 1}round train start---")
        model.train()

        total_train_loss = 0
        total_train_psnr = 0
        total_train_ssim = 0
        for data in train_dataloader:
            noisy_images, clean_images = data
            noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)

            outputs = model(noisy_images)
            loss = 0.7 * criterion_l1(outputs, clean_images) + 0.3 * criterion_l2(outputs, clean_images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            total_train_psnr += compute_psnr(outputs, clean_images)

            total_train_ssim += compute_ssim(outputs, clean_images)

        train_losses.append(total_train_loss / len(train_dataloader))
        train_psnrs.append(total_train_psnr / len(train_dataloader))
        train_ssims.append(total_train_ssim / len(train_dataloader))

        # val
        model.eval()
        total_val_loss = 0
        total_val_psnr = 0
        total_val_ssim = 0
        with torch.no_grad():
            for data in val_dataloader:
                noisy_images, clean_images = data
                noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)

                outputs = model(noisy_images)

                loss = 0.7 * criterion_l1(outputs, clean_images) + 0.3 * criterion_l2(outputs, clean_images)
                total_val_loss += loss.item()

                total_val_psnr += compute_psnr(outputs, clean_images)

                total_val_ssim += compute_ssim(outputs, clean_images)

        # SSIM
        val_losses.append(total_val_loss / len(val_dataloader))
        val_psnrs.append(total_val_psnr / len(val_dataloader))
        val_ssims.append(total_val_ssim / len(val_dataloader))

        # save
        torch.save(model.state_dict(), save_path)
        print(f"The model has been saved to: {save_path}")

        # test
        total_test_loss = 0
        total_test_psnr = 0
        total_test_ssim = 0
        with torch.no_grad():
            for data in test_dataloader:
                noisy_images, clean_images = data
                noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)

                outputs = model(noisy_images)

                loss = 0.7 * criterion_l1(outputs, clean_images) + 0.3 * criterion_l2(outputs, clean_images)
                total_test_loss += loss.item()

                total_test_psnr += compute_psnr(outputs, clean_images)


                total_test_ssim += compute_ssim(outputs, clean_images)

        # 记录测试集损失和 SSIM
        test_losses.append(total_test_loss / len(test_dataloader))
        test_psnrs.append(total_test_psnr / len(test_dataloader))
        test_ssims.append(total_test_ssim / len(test_dataloader))

        print(f"{epoch + 1} round training Loss: {train_losses[-1]:.4f}, PSNR: {train_psnrs[-1]:.4f}, SSIM: {train_ssims[-1]:.4f}")
        print(f"{epoch + 1} round training Loss: {val_losses[-1]:.4f}, PSNR: {val_psnrs[-1]:.4f}, SSIM: {val_ssims[-1]:.4f}")
        print(f"{epoch + 1} round training Loss: {test_losses[-1]:.4f}, PSNR: {test_psnrs[-1]:.4f}, SSIM: {test_ssims[-1]:.4f}")

    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 8))

    # loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss", color='blue')
    plt.plot(epochs_range, val_losses, label="Validation Loss", color='green')
    plt.plot(epochs_range, test_losses, label="Test Loss", color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train, Validation, and Test Loss')
    plt.legend()

    # PSNR
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, train_psnrs, label="Train PSNR", color='blue')
    plt.plot(epochs_range, val_psnrs, label="Validation PSNR", color='green')
    plt.plot(epochs_range, test_psnrs, label="Test PSNR", color='red')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.title('Train, Validation, and Test PSNR')
    plt.legend()

    # SSIM
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, train_ssims, label="Train SSIM", color='blue')
    plt.plot(epochs_range, val_ssims, label="Validation SSIM", color='green')
    plt.plot(epochs_range, test_ssims, label="Test SSIM", color='red')
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.title('Train, Validation, and Test SSIM')
    plt.legend()

    plt.tight_layout()
    plt.show()
