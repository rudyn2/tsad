#!/bin/bash

docker run --name ad-trainer -d --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1 -v ~/cachefs/carla_v2:/home/tsad/dataset -it ad-trainer /bin/bash
