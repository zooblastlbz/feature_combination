#!/bin/bash
set -x

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
mpirun --hostfile /etc/mpi/hostfile --pernode -x PATH sh -c 'rm -f /dev/shm/nccl-*'


export WANDB_API_KEY="c091a3f754adb7c44dbca6252e7f35ee202b87ef"
# 针对8机64卡配置调整slots
#num_slots=8  # 每个节点8张卡
#sed -i "s/slots=[0-9]\+\$/slots=$num_slots/g" /etc/mpi/hostfile

# NCCL网络和调试配置
rm .deepspeed_env
cp /ytech_m2v5_hdd/workspace/kling_mm/libozhou/feature_combination/env_a800 /root/.deepspeed_env


HOSTFILE=/etc/mpi/hostfile

NNODES=4
NPROC_PER_NODE=8
DEEPSPEED_CONFIG=/ytech_m2v5_hdd/workspace/kling_mm/libozhou/feature_combination/zero2.json

# 🔥 导出 DEEPSPEED_CONFIG 环境变量，让训练代码能够读取
export DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG}

/ytech_m2v5_hdd/workspace/kling_mm/libozhou/miniconda3/envs/fc-new/bin/deepspeed \
  --hostfile ${HOSTFILE} \
  --num_nodes ${NNODES} \
  --num_gpus ${NPROC_PER_NODE} \
    train.py \
    -c /ytech_m2v5_hdd/workspace/kling_mm/libozhou/feature_combination/configs/adafusedit/qwen3-vl-4b-4machine.yaml \
    --deepspeed_config ${DEEPSPEED_CONFIG}