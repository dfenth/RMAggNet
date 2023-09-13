# Reed-Muller Aggregation Networks (RMAggNet)

This repository contains the code from the paper [Using Reed-Muller Codes for Classification with Rejection and Recovery]()

A dockerfile is provided which can be used with NVIDIA GPUs. Please follow the instructions on the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) for details on how to set up docker with GPU support.

Remember to use `git submodule update --init --recursive` to install the CCAT code. Alternatively the repo and all submodules can be installed with `git clone --recurse-submodules <url to RMAggNet>`

Code was run on an NVIDIA A100 80 GPU

Apologies for the mess! This will be cleaned up by Friday!