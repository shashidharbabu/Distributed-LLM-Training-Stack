FROM pytorch/pytorch:2.1.1-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-client \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

RUN pip install --upgrade pip \
    && pip install -e .[development] \
    && pip cache purge

ENV NCCL_ASYNC_ERROR_HANDLING=1 \
    TORCH_DISTRIBUTED_DEBUG=INFO \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["llmtrain"]
