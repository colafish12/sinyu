import os
import time
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

start_time = time.time()

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

image_size = (256, 256)
save_path = "unet_model.pth"

#model
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
        # encoder
        self.c1 = DoubleConv(3, 64, dropout)
        self.d1 = nn.MaxPool2d(2)
        self.c2 = DoubleConv(64, 128, dropout)
        self.d2 = nn.MaxPool2d(2)
        self.c3 = DoubleConv(128, 256, dropout)
        self.d3 = nn.MaxPool2d(2)
        self.c4 = DoubleConv(256, 512, dropout)
        self.d4 = nn.MaxPool2d(2)

        # decoder
        self.u1 = UpSample(512, 256)
        self.c5 = DoubleConv(256 + 256, 256, dropout)

        self.u2 = UpSample(256, 128)
        self.c6 = DoubleConv(128 + 128, 128, dropout)

        self.u3 = UpSample(128, 64)
        self.c7 = DoubleConv(64 + 64, 64, dropout)

        self.c8 = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # encoder
        enc1 = self.c1(x)
        enc2 = self.c2(self.d1(enc1))
        enc3 = self.c3(self.d2(enc2))
        enc4 = self.c4(self.d3(enc3))

        # decoder
        dec1 = self.c5(torch.cat([self.u1(enc4), enc3], dim=1))
        dec2 = self.c6(torch.cat([self.u2(dec1), enc2], dim=1))
        dec3 = self.c7(torch.cat([self.u3(dec2), enc1], dim=1))

        # output
        output = self.c8(dec3)
        return output

model = UNet().to(device)

def load_trained_model(model, save_path, device):
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    print(f"Model loaded from {save_path}")


# denoise
def denoise_image(model, image_path, device):

    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    image = image.resize(image_size, Image.Resampling.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),transforms.Normalize(mean=[0.5,0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        denoised_image_tensor = model(image_tensor)

    denoised_image = denoised_image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    denoised_image = (denoised_image + 1) / 2
    denoised_image = np.clip(denoised_image, 0, 1)

    denoised_image = np.array(denoised_image)
    denoised_image = Image.fromarray((denoised_image * 255).astype(np.uint8))
    denoised_image = denoised_image.resize(original_size, Image.Resampling.LANCZOS)

    plt.imshow(denoised_image)
    plt.axis('off')
    plt.draw()

    end_time = time.time()
    total_time = end_time - start_time
    print(f"total time: {total_time:.4f} seconds")

    plt.show()
    denoised_image.save("denoised_image.jpg")

if __name__ == "__main__":
    load_trained_model(model, save_path, device)
    denoise_image(model, "C:\\Users\\Administrator\\Desktop\\113044.jpg", device)
