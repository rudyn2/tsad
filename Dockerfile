FROM anibali/pytorch:1.5.0-cuda10.2
USER root


# RUN conda env create -f environment.yml
# RUN apt-get install -y gnupg2
# RUN echo "deb http://us.archive.ubuntu.com/ubuntu/ bionic main" >> /etc/apt/sources.list
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32
# RUN apt-get update
RUN apt-get install libjpeg-turbo8 -y

RUN mkdir /home/tsad
WORKDIR /home/tsad
COPY requirements.txt /home/tsad/requirements.txt
RUN apt-get update
RUN pip install -r requirements.txt

# Copy code
COPY . /home/tsad/
COPY carla_egg/carla-0.9.11-py3.7-linux-x86_64.egg /home/tsad/carla_egg/
#COPY scripts/ /home/tsad/scripts/
#COPY models/ /home/tsad/models/
#COPY utils/ /home/tsad/utils/
RUN mkdir /home/tsad/dataset
WORKDIR /home/tsad/scripts


# Set some envirnonment variables
ENV PYTHONPATH "${PYTHONPATH}:/home/tsad/carla_egg/carla-0.9.11-py3.7-linux-x86_64.egg"

# RUN python scripts/train.py
