#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=72:00:00

# set name of job
#SBATCH -J ibp_bnn_fmnist_ablation

# set number of GPUs
#SBATCH --gres=gpu:2

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=skessler@robots.ox.ac.uk
python hibp_weight_pruning.py --dataset=fmnist --runs=5 --log_dir=logs_wp --tag=wp_ibp_fmnist > wp_ibp_fmnist.log