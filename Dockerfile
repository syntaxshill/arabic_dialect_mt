FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

## for apt to be noninteractive
ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

## Required packages
RUN apt update
RUN apt-get install -y curl wget git-core gcc make zlib1g zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libssl-dev libffi-dev lzma liblzma-dev libbz2-dev

## Python
RUN curl https://pyenv.run | bash

ENV HOME "/root"
ENV PYENV_ROOT "$HOME/.pyenv"
ENV PATH "${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"

RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
RUN echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc

RUN pyenv install 3.10.13
RUN pyenv global 3.10.13

ADD requirements.txt /opt/

# install requirements
RUN pip install -r /opt/requirements.txt
