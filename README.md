# COMP3359 CNN-based network of classifying Tropical Cyclone Satellite Image by Intensity

This repo contains code and notebooks of our machine learning project. Current it achieve around 70% accuracy.

## Introduction

In this project, we intend to develop a CNN-based model that accepts a satellite image of a tropical cyclone (TC) as input and predicts a label, an intensity class, for it. We intend to let an AI model perform it without knowing the exact rules and features to look for. There are six classes, namely class 0, class 1, class 2 a tropical depression, class 3 tropical storm, class 4 severe tropical storm and class 5 typhoon/hurricane (depending on in which part of the world do you live).

## Directory structure

    |- Demonstration.ipynb # Notebook of demo how to using trained model for prediction
    |- start_train.sh      # Entry point of sending batch to slurm manager
    |- train.py            # Python script of training a model when using sbatch
    |- Training.ipynb      # Main notebook covering data exploration, analysis, modeling,
    |                        training and trained model analysis
    |- typhoon_scraper.py  # Simple web scraper to get images from digital image (Deprecated)

## Installation Guide

> Please use Python 3.7 Conda  
> Linux is highly recommended, Untested in other platform

Optional. Create a conda virtual env

### Using Notebook

Follow instruction on notebook will do

### Using train.py

1. Install a proper version of pytorch

        # If you are using CUDA 10.0 (HKU GPU Phrase 1)
        # !pip install Pillow==6.1
        # !pip install torch===1.2.0 torchvision===0.4.0 -f https://download.pytorch.org/whl/torch_stable.html

        # If you have CUDA 10.1 (HKU GPU Phrase 2)
        # !pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

        # If you have CUDA 10.2
        # !pip install pytorch torchvision`

2. Install these packages by pip/conda `h5py sklearn efficientnet_pytorch`
