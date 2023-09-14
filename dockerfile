FROM pytorch/pytorch:latest

RUN apt update
RUN apt get-install -y git

# Install relevant extra packages
RUN pip install h5py
RUN pip install torchvision
RUN pip install scikit-learn
RUN pip install foolbox
RUN pip install matplotlib

COPY /src /code

WORKDIR /code

ENTRYPOINT [ "python3", "main.py" ]

# docker build -f docker_test -t dockertest .
# docker run --rm --gpus all -it dockertest sh # For interactive
# docker run --rm --gpus all -v $PWD/src:/code dockertest # No need if we copy files in dockerfile

# sudo docker build -f dockerfile -t rmaggnet .
# sudo docker run --rm --gpus all rmaggnet