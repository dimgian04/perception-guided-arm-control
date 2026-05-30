FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive

# Basic tools + Python
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    python3-numpy \
    python3-scipy \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Pinocchio πρώτα
RUN apt-get update && apt-get install -y \
    ros-humble-pinocchio \
    && rm -rf /var/lib/apt/lists/*

# ROS2 packages
RUN apt-get update && apt-get install -y \
    ros-humble-cv-bridge \
    ros-humble-vision-opencv \
    ros-humble-urdf \
    ros-humble-robot-state-publisher \
    && rm -rf /var/lib/apt/lists/*

# NumPy 1.x πρώτα, μετά opencv-contrib
RUN pip3 install "numpy<2" opencv-contrib-python

# Source ROS2 automatically
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

WORKDIR /workspace
