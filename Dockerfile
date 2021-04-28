FROM ubuntu:20.04
FROM pytorch/pytorch


# RUN conda env create -f environment.yml
# RUN apt-get install -y gnupg2
# RUN echo "deb http://us.archive.ubuntu.com/ubuntu/ bionic main" >> /etc/apt/sources.list
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32
# RUN apt-get update

RUN mkdir /home/tsad
WORKDIR /home/tsad
COPY requirements.txt /home/tsad/requirements.txt
RUN apt-get update
RUN pip install -r requirements.txt

# Copy code
COPY scripts/ /home/tsad/scripts/
COPY models/ /home/tsad/models/
COPY utils/ home/tsad/utils/
RUN mkdir /home/tsad/dataset
WORKDIR /home/tsad/scripts

# Set some envirnonment variables
# ENV PYTHONPATH "${PYTHONPATH}:/home/carla-dataset-runner/PythonAPI"
# ENV PYTHONPATH "${PYTHONPATH}:/home/carla-dataset-runner/carla_egg.egg"

# RUN python scripts/train.py
