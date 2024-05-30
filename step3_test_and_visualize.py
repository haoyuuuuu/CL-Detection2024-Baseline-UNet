"""
Project: CL-Detection2024 Challenge Baseline
============================================

This script implements cephalometric landmarks prediction on X-Ray images using a UNet-based heatmap approach.
It includes functionality for model testing, calculating MRE and SDR metrics, and visualizing the results.
此脚本实现了基于UNet热图的X射线图像头影关键点预测，包括模型测试、计算MRE和SDR指标以及结果可视化。

Email: zhanghongyuan2017@email.szu.edu.cn
"""

import tqdm
import torch
import argparse
import numpy as np
import pandas as pd
from skimage import transform
from skimage import io as sk_io

import warnings
warnings.filterwarnings('ignore')

from utils.model import load_model
from utils.cldetection_utils import check_and_make_dir, calculate_prediction_metrics, visualize_prediction_landmarks


def main(config):
    """
    Main function for model test and visualization.
    主函数，用于模型测试和可视化。
    :param config: Configuration object containing various configuration parameters.
                   包含各种配置参数的配置对象
    """
    # GPU device | GPU 设备
    gpu_id = config.cuda_id
    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

    # Load model | 加载模型
    model = load_model(model_name=config.model_name)
    model.load_state_dict(torch.load(config.load_weight_path, map_location=device))
    model = model.to(device)

    # Load test.csv | 加载测试数据集
    df = pd.read_csv(config.test_csv_path)

    # Test result dict | 测试结果字典
    test_result_dict = {}

    # Test mode | 测试模式
    with torch.no_grad():
        model.eval()
        # Test all images | 测试所有图片
        for index, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            image_file_path, spacing = str(df.iloc[index, 0]), float(df.iloc[index, 1])
            landmarks = df.iloc[index, 2:].values.astype('float')
            landmarks = landmarks.reshape(-1, 2)

            # Load image array | 加载图像数组
            image = sk_io.imread(image_file_path)
            h, w = image.shape[:2]
            new_h, new_w = config.image_height, config.image_width

            # Preprocessing image for model input | 为模型输入预处理图像
            image = transform.resize(image, (new_h, new_w), mode='constant', preserve_range=False)
            transpose_image = np.transpose(image, (2, 0, 1))
            torch_image = torch.from_numpy(transpose_image[np.newaxis, :, :, :]).float().to(device)

            # Predict heatmap | 预测热图
            heatmap = model(torch_image)

            # Transfer to landmarks | 转换为地标
            heatmap = np.squeeze(heatmap.cpu().numpy())
            predict_landmarks = []
            for i in range(np.shape(heatmap)[0]):
                landmark_heatmap = heatmap[i, :, :]
                yy, xx = np.where(landmark_heatmap == np.max(landmark_heatmap))
                # There may be multiple maximum positions, and a simple average is performed as the final result
                # 可能存在多个最大位置，并且进行简单平均以作为最终结果
                x0, y0 = np.mean(xx), np.mean(yy)
                # Zoom to original image size | 缩放到原始图像大小
                x0, y0 = x0 * w / new_w, y0 * h / new_h
                # Append to predict landmarks | 添加到预测地标
                predict_landmarks.append([x0, y0])

            test_result_dict[image_file_path] = {'spacing': spacing,
                                                 'gt': np.asarray(landmarks),
                                                 'predict': np.asarray(predict_landmarks)}

    # Calculate prediction metrics | 计算预测指标
    calculate_prediction_metrics(test_result_dict)

    # Visualize prediction landmarks | 可视化预测地标
    if config.save_image:
        check_and_make_dir(config.save_image_dir)
        visualize_prediction_landmarks(test_result_dict, config.save_image_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data parameters | 数据文件路径
    parser.add_argument('--test_csv_path', type=str, default='/data/XHY/CL-Detection2024/dataset/Training Set/test.csv')

    # Model load path | 模型路径
    parser.add_argument('--load_weight_path', type=str, default='/data/XHY/CL-Detection2024/model/baseline UNet/best_model.pt')

    # Model hyper-parameters: image_width and image_height
    # 模型超参数：图像宽度和高度
    parser.add_argument('--image_width', type=int, default=512)
    parser.add_argument('--image_height', type=int, default=512)

    # Model test hyper-parameters | 模型测试超参数
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='UNet')

    # Result & save | 结果和保存
    parser.add_argument('--save_image', type=bool, default=True)
    parser.add_argument('--save_image_dir', type=str, default='/data/XHY/CL-Detection2024/dataset/Training Set/visualize')

    experiment_config = parser.parse_args()
    main(experiment_config)

