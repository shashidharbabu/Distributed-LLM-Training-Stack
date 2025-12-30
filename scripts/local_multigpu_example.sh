#!/usr/bin/env bash
set -euo pipefail
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

# Launch training locally using torch multiprocessing (no torchrun dependency).
llmtrain run --gpus-per-node ${GPUS_PER_NODE} --num-nodes 1 --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT}
