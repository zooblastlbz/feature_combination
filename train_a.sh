export NCCL_TOPO_FILE=/share/baidu-nccl/topo_a800_hpc_bcc.xml
export LD_PRELOAD=/share/mayanqi/libnccl.so.2.27.5.ubuntu-cuda12.fix6
nohup deepspeed train.py -c configs/adafusedit/baseline.yaml > train.out 2>&1 &