# Rethinking Prompt-based Methods for Vision-Language Models

This is a project for 2022 Summer FIST course, Advanced Topics of Computer Vision

This is the code implementation of different prompts in zero-shot, few-shot and fully supervised tasks.

## Usage

First, [install PyTorch 1.7.1](https://pytorch.org/get-started/locally/) (or later) and torchvision, as well as small additional dependencies, and then install this repo as a Python package. On a CUDA GPU machine, the following will do the trick:

```bash
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install tqdm transformers
$ pip install git+https://github.com/openai/CLIP.git
```

Replace `cudatoolkit=11.0` above with the appropriate CUDA version on your machine or `cpuonly` when installing on a machine without a GPU.

## Dataset

The ImageNetV2 dataset contains new test data for the ImageNet benchmark.  You can download dataset by running the following command.

```bash
pip install git+https://github.com/modestyachts/ImageNetV2_pytorch
```