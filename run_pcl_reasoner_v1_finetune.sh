#!/bin/bash

export PYTHONPATH=/path/to/mindformers/:$PYTHONPATH

export MS_NODE_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export GLOG_v=2
export ACLNN_CACHE_LIMIT=0 # CANN 缓存限制
export MS_DEV_RUNTIME_CONF="memory_statistics:True,aclnn_cache_queue_length:0"
export MS_MEMORY_STATISTIC=1
export ENABLE_LAZY_INLINE=1
export MS_DEV_DUMP_IR_PASSES=graph_build

sysctl -w net.ipv4.ip_local_reserved_ports=60000-60015


noderank=$1 
bash /path/to/mindformers/scripts/msrun_launcher.sh "run_mindformer.py \
--config /path/to/finetune_qwen2_5_32b_32k.yaml \
--run_mode finetune" \
--worker_num 128 \
--local_worker_num 8 \
--master_addr XX.XX.XX.XX \
--master_port XXXX \
--node_rank $noderank \
--log_dir /path/to/log \
--join False \
--cluster_time_out 1200 \
> run.log 2>&1
