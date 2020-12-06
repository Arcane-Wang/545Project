#! /bin/bash
docker run -it --rm \
  -u $(id -u):$(id -g) \
  -v $PWD/logs:/logs:ro \
  -p 6006:6006 \
  completion-pc \
  tensorboard --logdir=runs --port=6006 --bind_all
