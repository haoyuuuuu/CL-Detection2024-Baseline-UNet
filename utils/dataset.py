"""
Project: CL-Detection2024 Challenge Baseline
============================================

Data Reading
数据读取

Email: zhanghongyuan2017@email.szu.edu.cn
"""

import pandas as pd
from skimage import io as sk_io
from torch.utils.data import Dataset

class CephXrayDataset(Dataset):
    def __init__(self, csv_file_path, transform=None, *args, **kwargs):
        """
        Initialize the CephXrayDataset with a CSV file containing image file paths and landmarks.
        使用包含图像文件路径和关键点的CSV文件初始化CephXrayDataset。
        Args:
            csv_file_path (str): Path to the CSV file containing image paths and landmarks.
                                 包含图像路径和关键点的CSV文件路径。
            transform (callable, optional): Optional transform to be applied on a sample.
                                            可选的转换应用于样本。
        """
        super().__init__(*args, **kwargs)
        self.landmarks_frame = pd.read_csv(csv_file_path)
        self.transform = transform

    def __getitem__(self, index):
        """
        Get a sample from the dataset at the specified index.
        获取指定索引的数据集样本。
        Args:
            index (int): Index of the sample to be fetched.
                         要获取的样本索引。
        Returns:
            sample (dict): A dictionary containing the image and its landmarks.
                           包含图像及其关键点的字典。
        """
        image_file_path = str(self.landmarks_frame.iloc[index, 0])
        image = sk_io.imread(image_file_path)
        landmarks = self.landmarks_frame.iloc[index, 2:].values.astype('float')
        landmarks = landmarks.reshape(-1, 2)

        sample = {'image': image, 'landmarks': landmarks}

        # Apply the transform if provided | 如果提供转换则应用
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        返回数据集中样本的总数。
        """
        return len(self.landmarks_frame)

