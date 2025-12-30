#!/usr/bin/env bash
set -euo pipefail
NUM_NODES=${NUM_NODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
NODE_RANK=${NODE_RANK:?must set NODE_RANK for this node}
MASTER_ADDR=${MASTER_ADDR:-llmtrain-headless}
MASTER_PORT=${MASTER_PORT:-29500}

export MASTER_ADDR MASTER_PORT
export NODE_RANK
export WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))

# Example invocation inside each node (e.g., Kubernetes pod)
llmtrain run --num-nodes ${NUM_NODES} --gpus-per-node ${GPUS_PER_NODE} --node-rank ${NODE_RANK} --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT}
