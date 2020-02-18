# Hierarchical Indian Buffet Neural Networks for Continual Learning

This code base runs the experiments in the ICML submission: Heirarchical Indian Buffet Neural Networks for Continual Learning.

The Continual Learning (CL) framework is based from [VCL](https://github.com/nvcuong/variational-continual-learning/tree/master/ddm). 
Structured VI of the IBP and H-IBP approximate posteriors is performed and loosely based off of an implementation from the [S-IBP VAE](https://github.com/rachtsingh/ibp_vae).

# Requirements

The requirements for a conda environments are listed in [requirements.txt](requirements.txt), together with setup instructions.

# Data

MNIST is included. MNIST variants are downloaded with the setup script, please run:
 
`bash setup.sh`

# Running the experiments

To see how to run each script and which arguments may be passed, run:

`cd src`
`python <script_name>.py -h`

The scripts below output a pickle file with the results. The notebook results.ipynb loads the pickle file for the permuted MNIST experiment below and plots the results.

## Permuted MNIST

To run the permuted MNIST experiments from the `src` directory:

`python run_permuted.py --num_layers=1 --run_baselines --h 5 50 100 --K 100 --tag=ibp > perm_ibp.log`

## Weight pruning

To run a simple weight pruning experiment, from the `src` directory run:

`python hibp_weight_pruning.py --run_baselines --hibp --tag=hibp_wp > hibp_wp.log`

## Split MNIST and Variants

To run the split MNIST experiments on the background images variant, from the `src` directory run:

`python run_split.py --num_layers=1 dataset=images --run_baselines --h 5 50 100 --tag=ibp > split_ibp.log`