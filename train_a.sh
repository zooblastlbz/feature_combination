#!/bin/bash
set -x

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
mpirun --hostfile /etc/mpi/hostfile --pernode -x PATH sh -c 'rm -f /dev/shm/nccl-*'

export WANDB_API_KEY="c091a3f754adb7c44dbca6252e7f35ee202b87ef"

# ===== Accelerate + DeepSpeed ZeRO-2 多机多卡配置 =====
NNODES=4
NPROC_PER_NODE=8
MIXED_PRECISION="bf16"

# 从 hostfile 获取主节点地址
HOSTFILE=/etc/mpi/hostfile
MASTER_ADDR=$(head -n 1 ${HOSTFILE} | cut -d' ' -f1)
MASTER_PORT=29500

# 配置文件路径
CONFIG_FILE=/ytech_m2v5_hdd/workspace/kling_mm/libozhou/feature_combination/configs/adafusedit/qwen3-vl-4b-4machine.yaml
ACCELERATE_CONFIG=/ytech_m2v5_hdd/workspace/kling_mm/libozhou/feature_combination/accelerate_config.yaml

# Python 环境
PYTHON_BIN=/ytech_m2v5_hdd/workspace/kling_mm/libozhou/miniconda3/envs/fc-new/bin

# 使用 accelerate launch + DeepSpeed ZeRO-2 启动多机多卡训练
${PYTHON_BIN}/accelerate launch \
    --config_file ${ACCELERATE_CONFIG} \
    --num_machines ${NNODES} \
    --num_processes $((NNODES * NPROC_PER_NODE)) \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --machine_rank ${RANK:-0} \
    train.py \
    -c ${CONFIG_FILE}