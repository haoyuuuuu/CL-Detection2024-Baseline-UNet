#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

# 构建docker
./build.sh

# 生成一个随机的字符串，用作卷标（volume label）的后缀
VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)

# 设置算法镜像的内存限制为8GB。在Grand Challenge上，当前的内存限制是30GB，但可以在算法镜像设置中进行配置
MEM_LIMIT="8g"

# 创建了一个名为cldetection_alg_2024-output-$VOLUME_SUFFIX的Docker卷。
# 其中$VOLUME_SUFFIX是一个变量，它的值通过生成一个随机的32位哈希值来确定
docker volume create cldetection_alg_2024-output-$VOLUME_SUFFIX

# 这是固定的参数，还请不要改变，其主要功能为：在一系列限制的环境中运行名为cldetection_alg_2024的镜像，并将本地路径和卷挂载到容器中供其访问
# --network="none"：禁用容器的网络功能，即容器内部无法访问网络
# --cap-drop="ALL"：禁用容器中的所有特权功能
# --security-opt="no-new-privileges"：禁止在容器内启用新的特权
# --shm-size="128m"：设置共享内存的大小为128MB
# --pids-limit="256"：限制容器的进程数上限为256个
# --name="test_container"：容器命名
# -v /data/XHY/CL-Detection2024/dataset/Validation\ Set/images/:/input/images/lateral-dental-x-rays/：
# 将/data/XHY/CL-Detection2024/dataset/Validation Set/images/本地预测图像目录路径挂载到容器中的/input/images/lateral-dental-x-rays/目录
# -v cldetection_alg_2024-output-$VOLUME_SUFFIX:/output/：将名为cldetection_alg_2024-output-$VOLUME_SUFFIX的Docker卷挂载到容器中的/output/目录
# cldetection_alg_2024：指定要运行的镜像名称，如果之前的 ./build.sh 修改了镜像名词，这里还请同步修改，建议不修改
docker run --gpus all \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        --name="test_container" \
        -v /data/XHY/CL-Detection2024/dataset/Validation\ Set/images/:/input/images/lateral-dental-x-rays/ \
        -v cldetection_alg_2024-output-$VOLUME_SUFFIX:/output/ \
        cldetection_alg_2024

# 复制容器内预测文件到主机路径
docker cp test_container:/output/predictions.csv /data/XHY/CL-Detection2024/dataset/Validation\ Set/predictions_docker.csv

# 删除容器
docker rm test_container

# 删除卷
docker volume rm cldetection_alg_2024-output-$VOLUME_SUFFIX
