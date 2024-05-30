"""
Project: CL-Detection2024 Challenge Baseline
============================================

This script utilizes the DetectionAlgorithm class to make predictions using a trained model,
and it saves the prediction results as a CSV file.
此脚本基于DetectionAlgorithm类对训练好的模型进行预测，并保存预测结果为csv文件

Email: xiehaoyu2022@email.szu.edu.cn
"""

import os.path
import SimpleITK
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from skimage import transform as sk_transform
from evalutils import DetectionAlgorithm
from evalutils.validators import UniquePathIndicesValidator, UniqueImagesValidator
from model import load_model

class Cldetection_alg_2023(DetectionAlgorithm):

    def __init__(self):
        # Please do not modify the initialization function of the parent class.
        # 请不要修改初始化父类的函数。
        super().__init__(
            validators=dict(input_image=(UniqueImagesValidator(), UniquePathIndicesValidator())),
            input_path=Path("/input/images/lateral-dental-x-rays/"),
            output_file=Path("/output/predictions.csv"))

        print("==> Starting...")

        # Use the corresponding GPU | 使用对应的GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model and weights. The path to the weights file is /opt/algorithm/best_model.pt.
        # 加载模型和权重，这里的权重文件路径是 /opt/algorithm/best_model.pt。
        # In the Docker environment, the current directory is mounted as /opt/algorithm/.
        # Therefore, any file in the current folder can be referenced in the code with the path /opt/algorithm/.
        # 在docker中会将当前目录挂载为 /opt/algorithm/，当前文件夹的任何文件在代码中的路径是 /opt/algorithm/。
        self.model = load_model(model_name='UNet')
        model_weight_path = '/opt/algorithm/best_model.pt'
        self.model.load_state_dict(torch.load(model_weight_path, map_location=self.device))
        self.model.to(self.device)

        print("==> Using ", self.device)
        print("==> Initializing model")
        print("==> Weights loaded")

    def save(self):
        """
        TODO: Rewrite the parent class function to save the results based on your own predict() function return type in self._output_file.
        TODO: 重写父类函数，根据自己的 predict() 函数返回类型，将结果保存在 self._output_file 中。
        """
        # All prediction results | 所有的预测结果
        all_images_predict_landmarks_list = self._case_results
        print("==> Predicted")

        # Save the prediction results in the CSV file format required for the challenge.
        # 预测结果转为挑战赛需要的CSV文件格式保存。
        columns = ['image file']
        for i in range(53):
            columns.extend(['p{}x'.format(i + 1), 'p{}y'.format(i + 1)])
        df = pd.DataFrame(columns=columns)

        # Iterate through each dictionary and write the data to a CSV file.
        # 遍历每个字典并将数据写入 CSV 文件
        for item in all_images_predict_landmarks_list:
            file_name = item['file name']
            landmarks = item['predict landmarks']
            row_line = [file_name] + [coord for point in landmarks for coord in point]
            df.loc[len(df.index)] = row_line

        df.to_csv(self._output_file, index=False)
        print("==> Saved CSV file")

    def process_case(self, *, idx, case):
        """
        !IMPORTANT: Please do not modify any content of this function. Below are the specific comments.
        !IMPORTANT: 请不要修改这个函数的任何内容，下面是具体的注释信息。
        """
        # Call the parent class's loading function | 调用父类的加载函数
        input_image, input_image_file_path = self._load_input_image(case=case)

        # Pass the corresponding input_image in SimpleITK.Image format and return the prediction results.
        # 传入对应的 input_image SimpleITK.Image 格式，返回预测结果。
        predict_one_image_result = self.predict(input_image=input_image, file_name=os.path.basename(input_image_file_path))

        return predict_one_image_result

    def predict(self, *, input_image: SimpleITK.Image, file_name: str):
        """
        TODO: Please modify the logic here to execute predictions using your own designed model. The return value can be in any format.
        TODO: 请修改这里的逻辑，执行自己设计的模型预测，返回值可以是任何形式的。
        :param input_image: The image that need to be predicted | 需要进行预测的图像
        :param file_name: The file name of the image | 图像文件名称
        :return: A dictionary consisting of file names and corresponding prediction values。
                文件名和预测值组成的字典。
        """
        # Convert the SimpleITK.Image format to Numpy.ndarray format for processing.
        # 将 SimpleITK.Image 格式转为 Numpy.ndarray 格式进行处理。
        image_array = SimpleITK.GetArrayFromImage(input_image)

        # Predict | 预测
        with torch.no_grad():
            self.model.eval()

            # Image preprocessing operations | 图片的预处理操作
            torch_image, image_info_dict = self.preprocess_one_image(image_array)

            # Model prediction | 模型的预测
            predict_heatmap = self.model(torch_image)

            # Post-processing of the results | 结果的后处理
            predict_landmarks = self.postprocess_model_prediction(predict_heatmap, image_info_dict=image_info_dict)

        return {'file name': file_name, 'predict landmarks': predict_landmarks}

    def preprocess_one_image(self, image_array: np.ndarray):
        """
        TODO: This is a custom function designed for preprocessing operations before inputting each image.
        TODO：这是一个自定义的函数，功能是对每一张图片输入前的预处理操作
        :param image_array: The data that need to be predicted | 需要进行预测的数据
        :return: The preprocessed image tensor and image information dictionary | 预处理后的图像张量和图像信息字典
        """
        # Basic information about the image | 图像的基本信息
        image_info_dict = {'width': np.shape(image_array)[1], 'height': np.shape(image_array)[0]}

        # Scale the image | 缩放图像
        scaled_image_array = sk_transform.resize(image_array, (512, 512), mode='constant', preserve_range=False)

        # Adjust the channel position, add a batch-size dimension, and convert to the torch format.
        # 调整通道位置，增加一个batch-size格式，并转为torch格式。
        transpose_image_array = np.transpose(scaled_image_array, (2, 0, 1))
        torch_image = torch.from_numpy(transpose_image_array[np.newaxis, :, :, :])

        # Move to a specific device | 转到特定的device上
        torch_image = torch_image.float().to(self.device)

        return torch_image, image_info_dict

    def postprocess_model_prediction(self, predict_heatmap: torch.Tensor, image_info_dict: dict):
        """
        TODO: Decode the model's prediction results to obtain the predicted coordinates of keypoints.
        TODO：对模型的预测结果进行解码，得到关键点的预测坐标值
        :param predict_heatmap: The predicted heatmap tensor from the model | 模型预测的热图张量
        :param image_info_dict: Information about the input image | 输入图像的信息
        :return: A list of predicted landmark coordinates | 预测的关键点坐标列表
        """
        # Obtain necessary image information for post-processing.
        # 得到一些必要的图像信息进行后处理。
        width, height = image_info_dict['width'], image_info_dict['height']

        # Convert to a Numpy matrix for processing: detach gradients, move to CPU, and convert to Numpy.
        # 转为Numpy矩阵进行处理: 去除梯度，转为CPU，转为Numpy。
        predict_heatmap = predict_heatmap.detach().cpu().numpy()

        # Remove the first batch-size dimension | 去除第一个batch-size维度
        predict_heatmap = np.squeeze(predict_heatmap)

        # Iterate through different heatmap channels to obtain the final output values.
        # 遍历不同的热图通道，得到最后的输出值。
        landmarks_list = []
        for i in range(np.shape(predict_heatmap)[0]):
            # Index to obtain different keypoint heatmaps.
            # 索引得到不同的关键点热图。
            landmark_heatmap = predict_heatmap[i, :, :]
            yy, xx = np.where(landmark_heatmap == np.max(landmark_heatmap))
            # There may be multiple maximum positions, and a simple average is performed as the final result
            # 可能有多个最大值，取平均值作为最终预测位置
            x0, y0 = np.mean(xx), np.mean(yy)
            # Zoom to original image size | 映射回原来图像尺寸
            x0, y0 = x0 * width / 512, y0 * height / 512
            # Append to landmarks list | 添加到关键点列表
            landmarks_list.append([x0, y0])

        return landmarks_list


if __name__ == "__main__":
    algorithm = Cldetection_alg_2023()
    algorithm.process()

    # Question: How can we call the process() function if it's not implemented here?
    # Answer: Because Cldetection_alg_2023 inherits from DetectionAlgorithm, it inherits the parent class's functions.
    #         So, when called, it automatically triggers the relevant functions.

    # 问：这里没有实现 process() 函数，怎么可以进行调用呢？
    # 答：因为这是 Cldetection_alg_2023 继承了 DetectionAlgorithm，父类函数，子类也就有了，然后进行执行，背后会自动调用相关函数

    # Question: What operations are performed behind the scenes when calling the process() function?
    # Answer: By referring to the source code, we can see the process() function, which is defined as follows:
    #    def process(self):
    #        self.load()
    #        self.validate()
    #        self.process_cases()
    #        self.save()
    #    We can see that these four functions are executed behind the scenes.
    #    Additionally, within the process_cases() function, the process_case() function is called:
    #    def process_cases(self, file_loader_key: Optional[str] = None):
    #        if file_loader_key is None:
    #            file_loader_key = self._index_key
    #        self._case_results = []
    #        for idx, case in self._cases[file_loader_key].iterrows():
    #            self._case_results.append(self.process_case(idx=idx, case=case))
    #    Therefore, you only need to implement the desired functionality in the process_case() and save() functions.

    # 问：调用 process() 函数，背后执行了什么操作呢？
    # 答：我们可通过跳转到源码可以看到，process() 函数，这里是源码显示：
    #    def process(self):
    #        self.load()
    #        self.validate()
    #        self.process_cases()
    #        self.save()
    #    我们可以看到背后执行了这四个函数，而对应在 process_cases() 函数中又进行了调用 process_case() 函数：
    #    def process_cases(self, file_loader_key: Optional[str] = None):
    #        if file_loader_key is None:
    #            file_loader_key = self._index_key
    #        self._case_results = []
    #        for idx, case in self._cases[file_loader_key].iterrows():
    #            self._case_results.append(self.process_case(idx=idx, case=case))
    #    因此，您仅需要在 process_case() 和 save() 函数中实现你想要的功能

    # Question: If only process_case() and save() need to be implemented, why is there also a predict() function?
    # Answer: The predict() function is required by the parent class DetectionAlgorithm to predict the results for each case.
    #         Otherwise, it would raise a NotImplementedError.

    # 问：又说仅需要 process_case() 和 save() 进行实现，为什么又跳出一个 predict() 函数呢？
    # 答：predict() 函数是父类 DetectionAlgorithm 要求实现的，负责预测每一个case的结果，不然会提示 NotImplementedError 错误



    
