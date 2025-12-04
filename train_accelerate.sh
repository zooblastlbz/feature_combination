#!/bin/bash

# Accelerate 训练启动脚本
# 使用方法: bash train_accelerate.sh configs/adafusedit/local_json_large.yaml

CONFIG_FILE=${1:-"configs/adafusedit/local_json_large.yaml"}
NUM_GPUS=${2:-8}
MIXED_PRECISION=${3:-"bf16"}

echo "🚀 启动 Accelerate 训练"
echo "  - 配置文件: $CONFIG_FILE"
echo "  - GPU 数量: $NUM_GPUS"
echo "  - 混合精度: $MIXED_PRECISION"

accelerate launch \
    --num_processes=$NUM_GPUS \
    --mixed_precision=$MIXED_PRECISION \
    --multi_gpu \
    train.py -c $CONFIG_FILE
