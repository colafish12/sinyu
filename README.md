# 图像去噪项目 - 基于 U-Net 网络

## 项目简介

本项目实现了一个基于 **U-Net** 深度学习模型的图像去噪任务。模型通过对输入的图像添加高斯噪声并使用去噪算法恢复图像的清晰度。项目使用 **PyTorch** 框架，提供了完整的训练、测试流程，并能进行图像去噪。

## 功能特点

- 使用 **U-Net** 网络架构，能够有效去除图像中的噪声。
- 训练过程中向原始图像添加高斯噪声，训练网络恢复清晰图像。
- 支持训练模型和对任意图像进行去噪处理。

**训练集**：将训练图像放置在 images/train/ 目录下，支持 .jpg 和 .png 格式。

**测试集**：将测试图像放置在 images/test/ 目录下。

要训练模型，运行以下命令
python Unet的搭建以及测试.py
该脚本会：
    加载训练集数据并添加高斯噪声。
    使用 U-Net 模型进行训练。
    每轮训练后会保存模型的权重到 unet_weights.pth 文件中。
    
训练轮数：默认为 100，可以根据需要调整。
批量大小：默认为 32，根据系统内存可调整。
学习率：默认为 1e-4。

训练完成之后，使用图片进行测试
python 图像测试.py

引号中的路径为你自己本地的图像路径
denoise_image(model, " ", device)

    
