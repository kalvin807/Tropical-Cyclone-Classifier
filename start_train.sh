#!/bin/bash

# Tell the system the resources you need. Adjust the numbers according to your need, e.g.
# SBATCH --gres=gpu:1 --cpus-per-task=4 --mail-type=ALL

#If you use Anaconda, initialize it
. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate base

# cd your your desired directory and execute your program, e.g.
python train.py
