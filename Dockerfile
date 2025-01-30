# Use CUDA 12.1-enabled base image for Ubuntu 22.04
# FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
# Continue with ROS setup from osrf/ros:humble-desktop-full
FROM osrf/ros:humble-desktop-full

ENV WS_DIR="/workspace"
WORKDIR ${WS_DIR}
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO humble
ENV DEBIAN_FRONTEND=noninteractive

# Set up timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# Install headers, CUDA repository key, and necessary tools
RUN apt-get update && \
    apt-get install -y gnupg2 curl ca-certificates && \
    rm -f /usr/share/keyrings/cuda-archive-keyring.gpg && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | gpg --dearmor --batch -o /usr/share/keyrings/cuda-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update

# Install CUDA and NVIDIA-related dependencies
RUN apt-get update && \
    apt-get install -y --allow-change-held-packages \
    cuda-toolkit-12-1 \
    libcudnn8 \
    libcudnn8-dev \
    && rm -rf /var/lib/apt/lists/*

# Remove any potential duplicate CUDA repository entries
RUN rm -f /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list

# Set up CUDA environment variables
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Install SO dependencies
RUN apt-get update -qq && \
    apt-get install -y \
    build-essential \
    cmake \
    git \
    libgtk2.0-dev \
    libgtk-3-dev \
    pkg-config \
    iputils-ping \
    wget \
    python3-pip \
    python3-dev \
    libtool \
    libpcap-dev \
    git-all \
    libeigen3-dev \
    libpcl-dev \
    software-properties-common \
    bash-completion \
    curl \
    tmux \
    zsh \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch, TorchVision, and Torchaudio
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Set up zsh with Oh My Zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended && \
    chsh -s $(which zsh)

# Set up a default .zshrc configuration with color support and green directory
RUN echo 'export TERM=xterm-256color' >> ~/.zshrc && \
    echo 'alias ll="ls -alF"' >> ~/.zshrc && \
    echo 'alias la="ls -A"' >> ~/.zshrc && \
    echo 'alias l="ls -CF"' >> ~/.zshrc && \
    echo 'export ZSH_THEME="robbyrussell"' >> ~/.zshrc && \
    echo 'PROMPT="%F{green}%~%f %F{blue}➜%f "' >> ~/.zshrc

# Set up tmux configuration
RUN echo 'set -g default-terminal "screen-256color"' >> ~/.tmux.conf && \
    echo 'set -g mouse on' >> ~/.tmux.conf

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# run with bash shell
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf && ldconfig

#run with zsh shell
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.zsh" >> ~/.zshrc
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf && ldconfig

# Default command to start bash shell interactively
CMD ["zsh"]
# CMD ["bash"] 