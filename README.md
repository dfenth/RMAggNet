# Reed-Muller Aggregation Networks (RMAggNet)

This repository contains the code from the paper *"Using Reed-Muller Codes for Classification with Rejection and Recovery"* ([long version](https://arxiv.org/abs/2309.06359)).

A dockerfile is provided which can be used with NVIDIA GPUs. Please follow the instructions on the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) for details on how to set up docker with GPU support.

Remember to use `git submodule update --init --recursive` to install the [CCAT](github.com/davidstutz/confidence-calibrated-adversarial-training) code. Alternatively the repo and all submodules can be installed with 
```
git clone --recurse-submodules https://github.com/dfenth/RMAggNet.git
```



**Apologies for the mess! This will be cleaned up by Friday!**


## Running code
The `main.py` file allows the code to be run in a modular fashion. There are two modes of operation, **training** which can be invoked with `--mode train` or **attacking** invoked with `--mode attack`. The only other required flag is `--dir` which specifies the directory to save any models to (in the case of `--mode train`) or load models from (in the case of `--mode attack`).

The command:
```
python3 main.py --mode train --dir my_folder
```
Will train all models over all datasets, saving them to the `my_folder` directory.

```
python3 main.py --mode attack --dir my_folder
```
Will attack all models with all of the available adversarial attacks over all datasets, with the models coming from the `my_folder` directory.

We can also make this more modular, specifying datasets, models and attacks (if applicable).

For instance, an example invocation:
```
python3 main.py --mode attack --dataset emnist --models ensemble --dir trained_models --boxtype openbox --attacks pgdlinf pgdl2 --cuda
```
Which performs open-box PGD Linf and L2 attacks on the Ensemble model trained on EMNIST using CUDA.

The options are:
- `--mode` - Defines the execution mode (`train` or `attack`) which trains the specified models or performs an adversarial attack
- `--dataset` - List all of the datasets to be included in the execution (select multiple from `mnist`, `emnist`, `cifar` or `all`)
- `--models` - List all of the models to be included in the run (select multiple from `rmaggnet`, `ensemble`, `ccat` or `all`)
- `--boxtype` - Specify the box type (select multiple from `openbox`, `closedbox` or `both`)
- `--attacks` - List all attacks to be used if `--mode attack` (select multiple from `pgdl2`, `pgdlinf`, `cwl2`, `boundary` or `all`)
- `--dir` - The directory to save the models to if `--mode train`, or load the models from if `--mode attack`
- `--cuda`/`--no-cuda` - Whether CUDA should be used for training/testing

## Running in docker
The repo includes a `dockerfile` which can be used to re-create the execution environment in docker. This can be built:
```
sudo docker build -f dockerfile -t rmaggnet .
```
And run using all of the command line options described above:
```
docker run --rm --gpus all rmaggnet --mode train --dataset mnist --models rmaggnet --dir newdir
```
Alternatively, it can be run in an interactive mode using:
```
sudo docker run --rm --gpus all -it rmaggnet sh
```
To use NVIDIA GPUs within the container for training/evaluation please see the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) webpage.

## Paper run environment
The code was run on an NVIDIA A100 80GB GPU. An 80 GB GPU seems necessary for the larger EMNIST and CIFAR-10 models because we do load a lot of models into GPU memory at once (and load a lot of the dataset as well). As a result around 16GB of RAM is also advised. I guarantee that this can be improved with some more careful memory management, but as often happens in research, we weren't able to get to this optimisation.

