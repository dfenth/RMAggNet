FROM pytorch/pytorch:latest

RUN apt update

# Install relevant extra packages
RUN pip install h5py
RUN pip install torchvision
RUN pip install scikit-learn
RUN pip install foolbox
RUN pip install matplotlib

COPY /src /code

WORKDIR /code

CMD python3 main.py

# docker build -f docker_test -t dockertest .
# docker run --rm --gpus all -it dockertest sh # For interactive
# docker run --rm --gpus all -v $PWD/src:/code dockertest # No need if we copy files in dockerfile