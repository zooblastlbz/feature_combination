#!/bin/bash
set -x

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
mpirun --hostfile /etc/mpi/hostfile --pernode -x PATH sh -c 'rm -f /dev/shm/nccl-*'

export WANDB_API_KEY="c091a3f754adb7c44dbca6252e7f35ee202b87ef"

rm -f .deepspeed_env
cp /ytech_m2v5_hdd/workspace/kling_mm/libozhou/feature_combination/env_h800 /root/.deepspeed_env

# 从 hostfile 获取主节点 IP
HOSTFILE=/etc/mpi/hostfile
MASTER_ADDR=$(head -n 1 ${HOSTFILE} | awk '{print $1}')
MASTER_PORT=29500

echo "🚀 Master Address: ${MASTER_ADDR}:${MASTER_PORT}"

# 配置文件路径
CONFIG_FILE=/ytech_m2v8_hdd/workspace/kling_mm/libozhou/feature_combination/configs/adafusedit/qwen3-vl-4b.yaml
ACCELERATE_CONFIG=/ytech_m2v8_hdd/workspace/kling_mm/libozhou/feature_combination/accelerate_config.yaml

# Python 环境
PYTHON_BIN=/ytech_m2v5_hdd/workspace/kling_mm/libozhou/miniconda3/envs/fc-new/bin

# 使用 accelerate launch + DeepSpeed ZeRO-2 启动多机多卡训练
# --main_process_ip 和 --main_process_port 会覆盖配置文件中的值
${PYTHON_BIN}/accelerate launch \
    --config_file ${ACCELERATE_CONFIG} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    train.py \
    -c ${CONFIG_FILE}