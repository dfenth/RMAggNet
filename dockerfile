FROM pytorch/pytorch:latest

RUN apt update

CMD python3 /code/adversarial_mnist_tests.py

# docker build -f docker_test -t dockertest .
# docker run --rm --gpus all -it dockertest sh # For interactive
# docekr run --rm --gpus all -v $PWD/rmaggnetcode:/code dockertest