"""
Project: CL-Detection2024 Challenge Baseline
============================================

This script implements cephalometric landmark detection on X-Ray images using a UNet-based heatmap approach.
It includes functionality for model training and validation.
此脚本基于UNet热图实现X-Ray图像的头影关键点定位，包括模型的训练和验证。

Email: zhanghongyuan2017@email.szu.edu.cn
"""

import os
import tqdm
import torch
import argparse
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')

from utils.tranforms import Rescale, RandomHorizontalFlip, ToTensor
from utils.dataset import CephXrayDataset
from utils.model import load_model
from utils.losses import load_loss

from utils.cldetection_utils import check_and_make_dir

def main(config):
    """
    Main function for model training and validation.
    主函数，用于模型训练和验证。
    :param config: Configuration object containing various configuration parameters.
                   包含各种配置参数的配置对象
    """
    # GPU device | GPU设备
    gpu_id = config.cuda_id
    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

    # Train and validation dataset | 训练和验证数据集
    train_transform = transforms.Compose([Rescale(output_size=(config.image_height, config.image_width)),
                                          RandomHorizontalFlip(p=config.flip_augmentation_prob),
                                          ToTensor()])
    train_dataset = CephXrayDataset(csv_file_path=config.train_csv_path, transform=train_transform)

    valid_transform = transforms.Compose([Rescale(output_size=(config.image_height, config.image_width)),
                                          ToTensor()])
    valid_dataset = CephXrayDataset(csv_file_path=config.valid_csv_path, transform=valid_transform)

    # Train and validation data loader | 训练和验证数据加载器
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.num_workers)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=config.batch_size_valid,
                              shuffle=False,
                              num_workers=config.num_workers)

    # Load model | 加载模型
    model = load_model(model_name=config.model_name)
    model = model.to(device)

    # Optimizer and StepLR scheduler | 优化器和StepLR调度器
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.lr,
                                 betas=(config.beta1, config.beta2))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=config.scheduler_step_size,
                                                gamma=config.scheduler_gamma)

    # Model loss function | 模型损失函数
    loss_fn = load_loss(loss_name=config.loss_name)

    # Model training preparation | 模型训练准备
    train_losses = []  # 用于存储每个epoch的训练损失
    valid_losses = []  # 用于存储每个epoch的验证损失
    best_loss = 1e10  # 初始最佳损失设置为一个较大的数
    num_epoch_no_improvement = 0  # 记录未提升的epoch次数
    check_and_make_dir(config.save_model_dir)  # 检查并创建模型保存文件夹

    # Start training and validation | 开始训练和验证
    for epoch in range(config.train_max_epoch):
        scheduler.step(epoch)
        model.train()
        for (image, heatmap) in tqdm.tqdm(train_loader):
            image, heatmap = image.float().to(device), heatmap.float().to(device)
            output = model(image)
            loss = loss_fn(output, heatmap)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(round(loss.item(), 3))
        print('Train epoch [{:<4d}/{:<4d}], Loss: {:.6f}'.format(epoch + 1, config.train_max_epoch, np.mean(train_losses)))

        # Save model checkpoint | 保存模型检查点
        if epoch % config.save_model_step == 0:
            torch.save(model.state_dict(), os.path.join(config.save_model_dir, 'checkpoint_epoch_%s.pt' % epoch))
            print("Saving checkpoint model ", os.path.join(config.save_model_dir, 'checkpoint_epoch_%s.pt' % epoch))

        # Validate model, save best model | 验证模型，保存最佳模型
        with torch.no_grad():
            model.eval()
            print("Validating....")
            for (image, heatmap) in tqdm.tqdm(valid_loader):
                image, heatmap = image.float().to(device), heatmap.float().to(device)
                output = model(image)
                loss = loss_fn(output, heatmap)
                valid_losses.append(loss.item())
        valid_loss = np.mean(valid_losses)
        print('Validation loss: {:.6f}'.format(valid_loss))

        # Early stopping mechanism | 早停机制
        if valid_loss < best_loss:
            print("Validation loss decreases from {:.6f} to {:.6f}".format(best_loss, valid_loss))
            best_loss = valid_loss
            num_epoch_no_improvement = 0
            torch.save(model.state_dict(), os.path.join(config.save_model_dir, "best_model.pt"))
            print("Saving best model ", os.path.join(config.save_model_dir, "best_model.pt"))
        else:
            print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss, num_epoch_no_improvement))
            num_epoch_no_improvement += 1
        if num_epoch_no_improvement == config.epoch_patience:
            print("Early Stopping!")
            break

        # Reset parameters | 重置参数
        train_losses = []
        valid_losses = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data parameters | 数据参数
    # Path Settings | 路径设置
    parser.add_argument('--train_csv_path', type=str, default='/data/XHY/CL-Detection2024/dataset/Training Set/train.csv')
    parser.add_argument('--valid_csv_path', type=str, default='/data/XHY/CL-Detection2024/dataset/Training Set/valid.csv')

    # Model hyperparameters: image_width and image_height | 模型超参数：image_width 和 image_height
    parser.add_argument('--image_width', type=int, default=512)
    parser.add_argument('--image_height', type=int, default=512)

    # Model training hyperparameters | 模型训练超参数
    parser.add_argument('--cuda_id', type=int, default=0)

    parser.add_argument('--model_name', type=str, default='UNet')
    parser.add_argument('--train_max_epoch', type=int, default=400)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--batch_size_valid', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--save_model_step', type=int, default=2)

    # Data augmentation | 数据增强
    parser.add_argument('--flip_augmentation_prob', type=float, default=0.5)

    # Model loss function | 模型损失函数
    parser.add_argument('--loss_name', type=str, default='focalLoss')

    # Early stopping mechanism | 早停机制
    parser.add_argument('--epoch_patience', type=int, default=5)

    # Learning rate | 学习率
    parser.add_argument('--lr', type=float, default=1e-4)

    # Adam optimizer parameters | Adam优化器参数
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Step scheduler parameters | Step调度器参数
    parser.add_argument('--scheduler_step_size', type=int, default=10)
    parser.add_argument('--scheduler_gamma', type=float, default=0.9)

    # Result & save | 结果和保存路径
    parser.add_argument('--save_model_dir', type=str, default='/data/XHY/CL-Detection2024/model/baseline UNet')

    experiment_config = parser.parse_args()
    main(experiment_config)

