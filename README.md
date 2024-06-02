# CL-Detection 2024 MICCAI Challenge Baseline UNet Model
This repository provides a solution based on UNet heatmap prediction for the [CL-Detection 2024 Challenge](https://www.codabench.org/competitions/2576/). Additionally, the repository includes a tutorial on packaging the solution as a Docker image, ensuring that participants can upload their algorithm models for validation on the leaderboard.

**NOTE:** The solution is built solely on the PyTorch framework without any additional framework dependencies (e.g., MMdetection). It contains detailed code comments for easy understanding and usage 🍚🍚🍚.

## Pipeline of This Solution 
The baseline solution provided in this repository is based on a paper published in the 2016 MICCAI conference: [Regressing Heatmaps for Multiple Landmark Localization Using CNNs](https://link.springer.com/chapter/10.1007/978-3-319-46723-8_27). The overall process of the designed approach is illustrated in the following diagram:

In the implementation provided in this repository, a UNet model is used as the backbone network for heatmap regression. The model takes input images of size 512×512 and outputs heatmaps with 53 channels. The predicted coordinates of the landmarks are obtained by taking the average coordinates of the maximum values in each heatmap.

## Reproducing the Experiment Results
Here is a step-by-step tutorial for you to follow. Alternatively, you can download the [pre-trained weight file](https://pan.baidu.com/s/13J39x9WM8smW8UyCp4GL8w?pwd=4jpz) and run step3_test_and_visualize.py to reproduce our results. Please remember to copy the downloaded weights to the step5_docker_and_upload folder for model inference in Docker.
 
How to run this code on the CL-Detection 2024 dataset or your own custom dataset?
This code can easily be executed for the landmark detection task. Here, we have divided the entire process into five steps to facilitate the reproduction of results based on the CL-Detection 2024 dataset or to execute the entire process on your custom dataset.

- Step0: Environment Setup.
- Step1: Run the script step1_dataset_split.py to split the public training set into three sets.
- Step2: Run the script step2_train_and_valid.py to train and validate the deep learning model.
- Step3: Run the script step3_test_and_visualize.py to test the model on test images.
- Step4: Run the script step4_predict_and_save.py to obtain the prediction results in csv format.
- Step5: Run the script step5_docker_and_upload to package the deep learning model.

**You should download the CL-Detection 2024 dataset in advance, following the detailed instructions provided on the challenge's official website.**

### Step0: Environment Setup
We have tested our code in following environment：

- Ubuntu == 18.04
- cuda == 11.3
- torch =＝1.11.0
- torchvision == 0.12.0

The repository code does not have any specific library dependencies. As long as you have torch and torchvision installed on your computer, you should be able to import them and install other dependencies. If you encounter any problems, please feel free to raise them in the Issues.

### Step1: Dataset Split
In Step 1, you should run the script step1_dataset_split.py in Python to perform the processing. This script handles the splitting of the dataset into distinct subsets: training, validation, and test sets. After running the code, you will obtain three CSV files, dividing the dataset into 300, 50, and 46 images, respectively, for training, testing, and validation purposes. The generated CSV files are organized in the following format: {image file name},{true physical distance of the pixel},{landmark X coordinate},{landmark Y coordinate}.

| image file | spacing(mm) | p1x | p1y | p2x | p2y | ... | p53x | p53y |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| 001.bmp | 0.1 | 835 | 996 | 1473 | 1029 | ... | 1712 | 1044 |
| 003.bmp | 0.1 | 761 | 1105 | 1335 | 923 | ... | 1577 | 861 |
| ┋ | ┋ | ┋ | ┋ | ┋ | ┋ | ┋ | ┋ | ┋ |

This repository follows a train-validate-test approach, where the model is trained on the training set, further trained and hyperparameters are selected on the validation set, and then tested on the test set to obtain the final model's performance.

**RUN:** 
- Modify the following input parameters and then run the script directly.
```python
# Path settings | 路径设置
parser.add_argument('--images_dir_path', type=str, default='/data/XHY/CL-Detection2024/dataset/Training Set/images',
                    help='Path to the images directory')
parser.add_argument('--labels_path', type=str, default='/data/XHY/CL-Detection2024/dataset/Training Set/labels.csv',
                    help='Path to the labels CSV file')
parser.add_argument('--split_dir_path', type=str, default='/data/XHY/CL-Detection2024/dataset/Training Set',
                    help='Path to the output split directory')
```
- Set the input parameters in the terminal and run it.
```python
python step1_dataset_split.py \
--images_dir_path='/data/XHY/CL-Detection2024/dataset/Training Set/images' \
--labels_path='/data/XHY/CL-Detection2024/dataset/Training Set/labels.csv' \
--split_dir_path='/data/XHY/CL-Detection2024/dataset/Training Set'
```

**NOTE:** The image preprocessing operations and dataset splitting mentioned above are not the only options. You are free to perform data processing and dataset splitting according to your preferences, or even expand it to a cross-validation mode for model training and validation. 

### Step2: Train and Valid
In Step 2, you can execute the script step2_train_and_valid.py to train models for predicting different landmarks. The train.csv file is used for model training, and the valid.csv file is used for validation. The training process utilizes an early stopping mechanism. The training stops when either the model's loss on the validation set does not decrease for a consecutive number of epochs (epoch_patience), or the maximum number of iterations (train_max_epoch) is reached. After executing this script and completing the training of the model, you will obtain a deep learning model capable of predicting heatmaps for 53 landmarks locations simultaneously.

**RUN:** 
- Modify the following input parameters and then run the script directly.
```python
# Data parameters | 数据参数
# Path Settings | 路径设置
parser.add_argument('--train_csv_path', type=str, default='/data/XHY/CL-Detection2024/dataset/Training Set/train.csv')
parser.add_argument('--valid_csv_path', type=str, default='/data/XHY/CL-Detection2024/dataset/Training Set/valid.csv')

# Model training hyperparameters | 模型训练超参数
parser.add_argument('--cuda_id', type=int, default=0)

parser.add_argument('--batch_size', type=int, default=8)

# Result & save | 结果和保存路径
parser.add_argument('--save_model_dir', type=str, default='/data/XHY/CL-Detection2024/model/baseline UNet')
```
- Set the input parameters in the terminal and run it.
```python
python step2_train_and_valid.py \
--train_csv_path='/data/XHY/CL-Detection2024/dataset/Training Set/train.csv' \
--valid_csv_path='/data/XHY/CL-Detection2024/dataset/Training Set/valid.csv' \
--cuda_id=0 \
--batch_size=8 \
--save_model_dir='/data/XHY/CL-Detection2024/model/baseline UNet'
```

### Step3: Test and Visualize
In Step 3, you should run the script step3_test_and_visualize.py to independently test the trained models and evaluate their performance. The script will assess the performance of all 53 landmarks on the entire set of images, which is the statistical approach used in the challenge. After running the script, you can observe the performance of the model on the independent test set in terms of the Mean Radial Error (MRE) and 2mm Success Detection Rate (SDR) metrics. The approximate values are MRE = 3.86 mm and 2mm SDR = 67.23%. Since the script does not fix the random seed, the results may have slight fluctuations within a small range, which could cause slight deviations from the experimental results provided by the authors.

**RUN:** 
- Modify the following input parameters and then run the script directly.
```python
# Data parameters | 数据文件路径
parser.add_argument('--test_csv_path', type=str, default='/data/XHY/CL-Detection2024/dataset/Training Set/test.csv')

# Model load path | 模型路径
parser.add_argument('--load_weight_path', type=str, default='/data/XHY/CL-Detection2024/model/baseline UNet/best_model.pt')

# Result & save | 结果和保存
parser.add_argument('--save_image', type=bool, default=True)
parser.add_argument('--save_image_dir', type=str, default='/data/XHY/CL-Detection2024/dataset/Training Set/visualize')
```
- Set the input parameters in the terminal and run it.
```python
python step3_test_and_visualize.py \
--test_csv_path='/data/XHY/CL-Detection2024/dataset/Training Set/test.csv' \
--load_weight_path='/data/XHY/CL-Detection2024/model/baseline UNet/best_model.pt' \
--save_image_dir='/data/XHY/CL-Detection2024/dataset/Training Set/visualize'
```

The following image shows the visualization of some test images from the test.csv file. The green dots represent the ground truth, i.e., the annotated landmarks by the doctors, while the red dots represent the model's predicted results. The yellow lines indicate the distances between the model's predictions and the doctor's annotations:

## Step4: Predict and Save
Test the model's predictions on the provided validation set images. After running the script, obtain the model's output results file named predictions.csv locally.

**RUN:** 
- Modify the following input parameters and then run the script directly.
```python
# Path Settings | 路径设置
parser.add_argument('--images_dir_path', type=str, default='/data/XHY/CL-Detection2024/dataset/Validation Set/images')
parser.add_argument('--save_csv_path', type=str, default='/data/XHY/CL-Detection2024/dataset/Validation Set/predictions.csv')
```
- Set the input parameters in the terminal and run it.
```python
python step4_predict_and_save.py \
--images_dir_path='/data/XHY/CL-Detection2024/dataset/Validation Set/images' \
--save_csv_path='/data/XHY/CL-Detection2024/dataset/Validation Set/predictions.csv' \
```

## Step5: Docker Upload
First, make sure that Docker and NVIDIA Container Toolkit are installed on your computing platform as they are essential for the algorithm packaging. The former ensures that you can perform the packaging, while the latter enables GPU utilization within Docker. Be sure to confirm that your system has been properly installed and configured.

Next, make sure to modify the requirements.txt file to include the necessary dependencies for your code project. This ensures that all the required libraries for the prediction process are included so that the prediction code can be executed correctly and produce the desired results.

Then, implement your inference testing process in the predict() function within the process.py file, and modify the save() function based on the return value of predict() function. It's important to note that there are no strict requirements for the return value of the predict() function, so you can structure it according to your programming preferences.

After that, execute the build.sh script to troubleshoot any errors. If everything goes smoothly, you may see the following output:
```python
=> exporting to image                                                                                                                                        0.2s
=> => exporting layers                                                                                                                                       0.2s
=> => writing image sha256:1b360361c1ea8a004f2e6c506e30fe0cd3d9be1806755283342e3f468f5a4d62                                                                  0.0s
=> => naming to docker.io/library/cldetection_alg_2024                                                                                                       0.0s
```
Finally, execute the test.sh script to verify if the output results from Docker match the locally predicted results. If they match, proceed to execute the export.sh script to export the CLdetection_Alg_2024.tar.gz file that can be uploaded to the challenge platform.

## Tips for Participants
This repository only provides a baseline model and a complete workflow for training, testing, and packaging for participants. The performance of the model is not very high, and the organizers may suggest the following directions for optimization as a reference:
- Design preprocessing and data augmentation strategies that are more targeted. This repository only involves simple image scaling to a size of (512, 512) and horizontal flipping for augmentation.
- Replace the backbone network with more powerful models such as the HRNet, Hourglass models, or Transformer models with self-attention mechanisms.
- Incorporate powerful attention modules. It is common in research to enhance model generalization and performance using attention mechanisms.
- Choosing a suitable loss function can make it easier for the deep learning model to learn and converge more quickly, leading to higher performance.
Finally, if you encounter any challenges or difficulties while participating in the CL-Detection 2024 challenge, encounter any errors while running the code in this repository, or have any suggestions for improving the baseline model, please feel free to raise an issue. I will be actively available to provide assistance and support.
