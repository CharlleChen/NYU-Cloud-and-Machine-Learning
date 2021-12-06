FROM ubuntu:18.04


# install utiliteis
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# create a directory for work
RUN mkdir /home/user
ENV HOME=/home/user
RUN chmod 777 /home/user

ENV PATH=/home/user/miniconda/bin:$PATH
# since docker is organized layer by layer, we incorporate multiple commands to decrease the number of layers
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.3-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p ~/miniconda \
    && rm ~/miniconda.sh \
    && conda install -y python==3.8.3

RUN conda install -y pytorch torchvision

RUN curl -o tiny-imagenet-200.zip http://cs231n.stanford.edu/tiny-imagenet-200.zip
RUN apt-get update && apt-get install -y unzip && unzip -q tiny-imagenet-200.zip

COPY train.py .

CMD python3 train.py
