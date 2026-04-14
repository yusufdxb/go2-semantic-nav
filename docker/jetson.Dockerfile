# Jetson deployment container for go2-semantic-nav.
# Base: NVIDIA L4T PyTorch for JetPack 6.x (Orin NX 16 GB).
#
# Build on the Jetson:
#   docker build -f docker/jetson.Dockerfile -t go2-semantic-nav:jetson .
#
# Run:
#   docker run --rm -it --runtime nvidia --network host \
#     -v "$PWD:/workspace" \
#     -v "$HOME/.cache:/home/ros/.cache" \
#     go2-semantic-nav:jetson
FROM nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.3-py3

ENV DEBIAN_FRONTEND=noninteractive
ARG ROS_DISTRO=humble

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg lsb-release software-properties-common \
    git build-essential cmake pkg-config \
    python3-pip python3-dev \
    libeigen3-dev \
 && rm -rf /var/lib/apt/lists/*

# --- ROS 2 Humble on Ubuntu 22.04 (Jammy, JetPack 6.x) ---
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2.list \
 && apt-get update && apt-get install -y --no-install-recommends \
    ros-${ROS_DISTRO}-ros-base \
    ros-${ROS_DISTRO}-nav2-bringup \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-tf2-ros \
    ros-${ROS_DISTRO}-vision-msgs \
    ros-${ROS_DISTRO}-rviz2 \
    python3-colcon-common-extensions \
 && rm -rf /var/lib/apt/lists/*

# --- Python ML stack ---
# IMPORTANT: torch is already in the base image; do NOT reinstall it — pip will
# replace the Jetson-optimized wheel with a CPU-only one.
RUN pip install --upgrade pip \
 && pip install \
    "numpy<2.0" \
    ultralytics>=8.4.0 \
    open_clip_torch>=2.24.0 \
    networkx>=3.2 \
    scipy>=1.11.0 \
    opencv-python>=4.8.0 \
    pyyaml \
    pytest \
 && pip install --no-binary=:all: sentencepiece

# MobileSAM from source (x86 wheel won't run on aarch64).
RUN pip install git+https://github.com/ChaoningZhang/MobileSAM.git

# NanoSAM (Jetson-native TRT segmenter).
RUN cd /opt && git clone --depth 1 https://github.com/NVIDIA-AI-IOT/nanosam.git \
 && cd nanosam && pip install -e .

# Default shell sources ROS 2 + any workspace mounted at /workspace/install.
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc \
 && echo "[ -f /workspace/ros2_ws/install/setup.bash ] && source /workspace/ros2_ws/install/setup.bash" >> /root/.bashrc \
 && echo "export OMP_NUM_THREADS=4" >> /root/.bashrc

WORKDIR /workspace
CMD ["/bin/bash"]
