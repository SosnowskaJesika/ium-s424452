FROM ubuntu:latest

RUN apt update && apt install -y figlet
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN python3 -m pip install pandas
RUN python3 -m pip install numpy
RUN python3 -m pip install torch
RUN python3 -m pip install torchvision

COPY train_model.py .
