#!/bin/bash

docker build -t ad-trainer .
docker run --name ad-trainer -d --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1 -it ad-trainer /bin/bash
