# refer:https://hub.docker.com/layers/nvidia/cuda/11.8.0-cudnn8-devel-ubuntu22.04/images/sha256-ee127a83b5269251476a3a1933972735a0a8a3269d35fda374f548212687d5db?context=explore
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV HOME /home
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/bin:$PATH

# refer:https://zenn.dev/efficientyk/articles/0fde4dcd4a9520
RUN apt-get update \
    && apt-get install -y \
    git \
    git-lfs \
    curl \
    wget \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libbz2-dev \
    libnss3-dev \
    libsqlite3-dev \
    libssl-dev \
    liblzma-dev \
    libreadline-dev \
    libffi-dev \
    libgl1-mesa-dev \
    locales \
    fish \
    vim \
    nano \
    iputils-ping \
    net-tools \
    software-properties-common \
    fonts-powerline

RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv && \
    echo 'eval "$(pyenv init --path)"' >> ~/.bashrc && \
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc

RUN pyenv install 3.11.6
RUN pyenv global 3.11.6