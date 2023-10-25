FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ARG WANDB_KEY

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# set the working directory and copy everything to the docker file
WORKDIR ./
COPY ./requirements.txt ./

RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install -y git
RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

RUN if [ -z "$WANDB_KEY" ] ; then echo WandB API key not provided ; else wandb login "$WANDB_KEY"; fi
