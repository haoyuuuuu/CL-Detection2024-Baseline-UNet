"""
Project: CL-Detection2024 Challenge Baseline
============================================

This script is used to make predictions using a pre-trained model and save the prediction results to a CSV file.
此脚本用于对训练好的模型进行预测，并保存预测结果为csv文件

Email: xiehaoyu2022@email.szu.edu.cn
"""

import os
import tqdm
import torch
import argparse
import numpy as np
import pandas as pd
from skimage import io as sk_io
from skimage import transform as sk_transform

import warnings
warnings.filterwarnings('ignore')

from utils.model import load_model

def main(config):
    """
    Main function for model testing and saving results
    主函数，用于模型测试和保存结果
    :param config: Configuration object containing various parameters
    包含各种配置参数的配置对象
    """
    # GPU device | GPU设备
    gpu_id = config.cuda_id
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

    # Load model | 加载模型
    model = load_model(model_name=config.model_name)
    model.load_state_dict(torch.load(config.load_weight_path, map_location=device))
    model = model.to(device)

    # Obtain image files list | 获取图像文件列表
    image_file_list = os.listdir(config.images_dir_path)
    image_file_list.sort()

    # CSV header | 表格列
    columns = ['image file']
    for i in range(53):
        columns.extend(['p{}x'.format(i + 1), 'p{}y'.format(i + 1)])
    df = pd.DataFrame(columns=columns)

    # Test mode | 测试模式
    with torch.no_grad():
        model.eval()
        for image_file_name in tqdm.tqdm(image_file_list):
            # Read image | 读取数据
            image = sk_io.imread(os.path.join(config.images_dir_path,image_file_name))
            height, width = np.shape(image)[:2]

            # Resize | 调整大小
            scaled_image = sk_transform.resize(image, (512, 512), mode='constant', preserve_range=False)

            # Transpose channel and add batch-size channel | 转换通道并添加批量大小通道
            transpose_image = np.transpose(scaled_image, (2, 0, 1))
            torch_image = torch.from_numpy(transpose_image[np.newaxis, :, :, :]).float().to(device)

            # Model predict | 模型预测
            predict_heatmap = model(torch_image)

            # Decode landmark location from heatmap | 从热图中解码关键点位置
            predict_heatmap = predict_heatmap.detach().cpu().numpy()
            predict_heatmap = np.squeeze(predict_heatmap)

            landmarks_list = []
            for i in range(np.shape(predict_heatmap)[0]):
                # Index different landmark heatmaps | 索引得到不同的关键点热图
                landmark_heatmap = predict_heatmap[i, :, :]
                yy, xx = np.where(landmark_heatmap == np.max(landmark_heatmap))
                # There may be multiple maximum positions, and a simple average is performed as the final result
                # 可能存在多个最大位置，并且进行简单平均以作为最终结果
                x0, y0 = np.mean(xx), np.mean(yy)
                # Zoom to original image size | 缩放到原始图像大小
                x0, y0 = x0 * width / 512, y0 * height / 512
                # Append to landmarks list | 添加到关键点列表
                landmarks_list.append([x0, y0])
            # Write prediction results | 预测结果写入
            row_line = [image_file_name]
            for i in range(53):
                point = landmarks_list[i]
                row_line.extend([point[0], point[1]])
            df.loc[len(df.index)] = row_line

    # CSV writer | CSV写入
    df.to_csv(config.save_csv_path, index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data parameters | 数据参数
    # Path Settings | 路径设置
    parser.add_argument('--images_dir_path', type=str, default='/data/XHY/CL-Detection2024/dataset/Validation Set/images')
    parser.add_argument('--save_csv_path', type=str, default='/data/XHY/CL-Detection2024/dataset/Validation Set/predictions.csv')

    # Model load path | 存放模型的文件路径
    parser.add_argument('--load_weight_path', type=str, default='/data/XHY/CL-Detection2024/model/baseline UNet/best_model.pt')

    # Model hyper-parameters: image_width and image_height
    # 模型超参数：图像宽度和高度
    parser.add_argument('--image_width', type=int, default=512)
    parser.add_argument('--image_height', type=int, default=512)

    # Model test hyper-parameters | 模型测试超参数
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='UNet')

    experiment_config = parser.parse_args()
    main(experiment_config)
