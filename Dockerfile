# syntax=docker/dockerfile:1

# $Source: /home/c/Dropbox/src/docker/dockertests/RCS/Dockerfile,v $
# $Date: 2022/12/31 16:15:49 $
# $Revision: 1.3 $

FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel as llmexp2

ARG O31_UBUNTU_RELEASE=22.04
ARG O31_UBUNTU_CODEJAME=jammy

SHELL ["/bin/bash", "-ceoux", "pipefail"]

ARG DEBIAN_FRONTEND=noninteractive
ARG MIRROR1=/var/cache/mirror
ARG WGET1="wget --directory-prefix=$MIRROR1 --force-directories --protocol-directories --timestamping"

# Disable automatic cache cleanup so we can persist the apt cache between build runs
# https://contains.dev/blog/mastering-docker-cache
RUN \
    --mount=type=cache,target=/root/.cache,sharing=locked \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/cache/mirror,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    <<EOF
  useradd -mp "" -s /bin/bash c
  mkdir -p /home/c/.cache/huggingface
  mkdir -p /tmp/offload
  chown c:c /tmp/offload
  rm -f /etc/apt/apt.conf.d/docker-clean
  echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
    > /etc/apt/apt.conf.d/keep-cache
  apt update -y && apt upgrade -y
EOF

COPY environment /etc/environment

RUN \
    --mount=type=cache,target=/root/.cache,sharing=locked \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/cache/mirror,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    <<EOF
  apt update -y && apt install -y \
    ack \
    git \
    lz4 \
    python3-{dev,pip,setuptools,venv} \
    rsync \
    tmux \
    tree \
    vim \
    wget \

  pip3 install \
    accelerate==0.15.0 \
    bitsandbytes==0.35.4 \
    ipython \
    transformers==4.25.1 \
    wandb==0.13.7 \

  $WGET1 \
    "https://www.christopheroei.com/b/7a25ad28e78ca414ccabf032a8fc8ef49b0bef0faf5ba6f3326888c610e04a16.sh" \

  cp "$MIRROR1/https/www.christopheroei.com/b/7a25ad28e78ca414ccabf032a8fc8ef49b0bef0faf5ba6f3326888c610e04a16.sh" \
    /etc/stringstack.sh
EOF

COPY tmux.conf /home/c/.tmux.conf
COPY exp2.py /home/c/

RUN <<EOF
  echo "export TERM=xterm-256color" >> /home/c/.bashrc
  chown -R c.c /home/c
EOF

EXPOSE 7860

# vim: set et ff=unix ft=dockerfile nocp sts=2 sw=2 ts=2:
