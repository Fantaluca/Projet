FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN echo 'Acquire::https::Verify-Peer "false";' > /etc/apt/apt.conf.d/99verify-peer.conf

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apt-transport-https \
        wget \
        gnupg \
        software-properties-common \
        git \
        build-essential \
        python3 \
        cmake \
        ninja-build \
        openmpi-bin \
        libopenmpi-dev \
        ca-certificates \
        curl \
        tree && \
    echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::http::Timeout "240";' > /etc/apt/apt.conf.d/80-timeouts && \
    echo 'Acquire::https::Timeout "240";' >> /etc/apt/apt.conf.d/80-timeouts && \
    curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/llvm-archive-keyring.gpg] https://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" > /etc/apt/sources.list.d/llvm.list && \
    for i in {1..3}; do apt-get update && break || sleep 15; done && \
    for i in {1..3}; do \
        apt-get install -y --no-install-recommends \
            clang-18 \
            lldb-18 \
            lld-18 \
            clangd-18 \
            libomp-18-dev \
            libclang-18-dev \
            clang-tools-18 && break || sleep 15; \
    done && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/lib/llvm-18/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/lib/llvm-18/lib:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=/usr/lib/llvm-18/lib:${LIBRARY_PATH}

# Configuration CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=${CUDA_HOME}/lib64:${LIBRARY_PATH}

# Configuration OpenMP offloading
ENV OMPI_CC=clang-18
ENV OMP_TARGET_OFFLOAD=MANDATORY
ENV LIBOMPTARGET_PLUGIN_PATH=/usr/lib/llvm-18/lib
ENV LIBOMPTARGET_DEVICETYPE=nvidia
ENV LIBOMPTARGET_NVPTX_ARCH=sm_89
ENV LIBOMPTARGET_CUDA_DEVICE=0
ENV LIBOMPTARGET_CUDA_JIT=1
ENV LIBOMPTARGET_STACK_SIZE=2048
ENV LIBOMPTARGET_HEAP_SIZE=2048

# Debug settings for OpenMP
ENV LIBOMPTARGET_DEBUG=1
ENV LIBOMPTARGET_INFO=4
ENV LIBOMPTARGET_PROFILE=1
ENV LIBOMPTARGET_PLUGIN_DEBUG=1
ENV LIBOMPTARGET_PLUGIN_PROFILE=1

# Configuration CUDA supp
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

WORKDIR /opt/hpsc_container

# docker build -t hpsc_image .
# docker run --gpus all -v "${PWD}:/app" -it hpsc_image bash