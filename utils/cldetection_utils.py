"""
Project: CL-Detection2024 Challenge Baseline
============================================

Custom functions
自定义函数

Email: zhanghongyuan2017@email.szu.edu.cn
"""

import os
import shutil
import numpy as np
from skimage import io as sk_io
from skimage import draw as sk_draw

def check_and_make_dir(dir_path: str) -> None:
    """
    function to create a new folder, if the folder path dir_path in does not exist
    :param dir_path: folder path | 文件夹路径
    :return: None | 无
    """
    if os.path.exists(dir_path):
        if os.path.isfile(dir_path):
            raise ValueError('Error, the provided path (%s) is a file path, not a folder path.' % dir_path)
    else:
        os.makedirs(dir_path)


def calculate_prediction_metrics(result_dict: dict):
    """
    function to calculate prediction metrics | 计算评价指标
    :param result_dict: a dict, which stores every image's predict result and its ground truth landmark
                        一个字典，存储每个图像的预测结果及其真实地标
    :return: MRE and 2mm SDR metrics | MRE 和 2mm SDR 指标
    """
    n_landmarks = 0
    sdr_landmarks = 0
    n_landmarks_error = 0
    for file_path, landmark_dict in result_dict.items():
        spacing = landmark_dict['spacing']
        landmarks, predict_landmarks = landmark_dict['gt'], landmark_dict['predict']

        # landmarks number
        n_landmarks = n_landmarks + np.shape(landmarks)[0]

        # mean radius error (MRE)
        each_landmark_error = np.sqrt(np.sum(np.square(landmarks - predict_landmarks), axis=1)) * spacing
        n_landmarks_error = n_landmarks_error + np.sum(each_landmark_error)

        # 2mm success detection rate (SDR)
        sdr_landmarks = sdr_landmarks + np.sum(each_landmark_error < 2)

    mean_radius_error = n_landmarks_error / n_landmarks
    sdr = sdr_landmarks / n_landmarks

    print('Mean Radius Error (MRE): {}, 2mm Success Detection Rate (SDR): {}'.format(mean_radius_error, sdr))
    return mean_radius_error, sdr


def visualize_prediction_landmarks(result_dict: dict, save_image_dir: str):
    """
    function to visualize prediction landmarks | 可视化预测结果
    :param result_dict: a dict, which stores every image's predict result and its ground truth landmark
                       一个字典，存储每个图像的预测结果及其真实地标
    :param save_image_dir: the folder path to save images | 一个字典，存储每个图像的预测结果及其真实地标
    :return: None | 无
    """
    for file_path, landmark_dict in result_dict.items():
        landmarks, predict_landmarks = landmark_dict['gt'], landmark_dict['predict']

        image = sk_io.imread(file_path)
        image_shape = np.shape(image)[:2]

        for i in range(np.shape(landmarks)[0]):
            landmark, predict_landmark = landmarks[i, :], predict_landmarks[i, :]
            # ground truth landmark
            radius = 7
            rr, cc = sk_draw.disk(center=(int(landmark[1]), int(landmark[0])), radius=radius, shape=image_shape)
            image[rr, cc, :] = [0, 255, 0]
            # model prediction landmark
            rr, cc = sk_draw.disk(center=(int(predict_landmark[1]), int(predict_landmark[0])), radius=radius, shape=image_shape)
            image[rr, cc, :] = [255, 0, 0]
            # the line between gt landmark and prediction landmark
            line_width = 5
            rr, cc, value = sk_draw.line_aa(int(landmark[1]), int(landmark[0]), int(predict_landmark[1]), int(predict_landmark[0]))
            for offset in range(line_width):
                offset_rr, offset_cc = np.clip(rr + offset, 0, image_shape[0] - 1), np.clip(cc + offset, 0, image_shape[1] - 1)
                image[offset_rr, offset_cc, :] = [255, 255, 0]

        filename = os.path.basename(file_path)
        sk_io.imsave(os.path.join(save_image_dir, filename), image)


