#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=04:00:00

# set name of job
#SBATCH -J rs_img_cl1

# set number of GPUs
#SBATCH --gres=gpu:4

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=skessler@robots.ox.ac.uk
#python weight_pruning.py --dataset=fmnist --runs=3 --no_ibp --run_baselines --log_dir=logs_wp --tag=wp_ibp_fmnist
python random_search.py --dataset=images --num_layers=1 --runs=20 --log_dir=logs_rs --K=100 --tag=ibp_img_cl1_rs
