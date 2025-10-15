FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ARG TARGETARCH
ENV DEBIAN_FRONTEND=noninteractive

#install developers pkg/repo
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
       #gcc needed for build_oai
       build-essential \
       psmisc \
       git \
       gh \
       gpg \
       #use gcc-12 to avoid problems of default gcc-11 in FlexRIC
       gdb \
       rr \
       python3-dev \
       # python3-pip for conf template generation
       python3-pip && \
    rm -rf /var/lib/apt/lists/*

USER ubuntu
# install astral uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/ubuntu/.local/bin:$PATH"

# Set working directory
WORKDIR /workspace