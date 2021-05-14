#!/bin/bash

docker run --name ad-trainer2 --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=5 -v ../rudy.garcia/cachefs/carla_v2_small_johnny:/home/tsad/dataset -it ad-trainer /bin/bash