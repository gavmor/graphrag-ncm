# Keep the historical dependency image unchanged; add task-level GNU time only.
FROM 127.0.0.1:5000/ncm-fewnerd-probe@sha256:a2cfb54d8a9da6517c4978f122d43176c06847440426728818e1226308659532
RUN apt-get update && apt-get install -y --no-install-recommends time \
    && rm -rf /var/lib/apt/lists/*
RUN CUDA_VISIBLE_DEVICES="" /usr/bin/time -v python -c "import torch, boto3, transformers; print(torch.__version__)"
