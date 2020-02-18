# Hierarchical Indian Buffet Neural Networks for Continual Learning

This code base runs the experiments in the ICML submission: Heirarchical Indian Buffet Neural Networks for Continual Learning.

The Continual Learning (CL) framework is based from [VCL](https://github.com/nvcuong/variational-continual-learning/tree/master/ddm). 
Structured VI of the IBP and H-IBP approximate posteriors is performed and loosely based off of an implementation from the [S-IBP VAE](https://github.com/rachtsingh/ibp_vae).

# Requirements

The requirements for a conda environments are listed in [requirements.txt](requirements.txt), together with setup instructions.

# Data

MNIST is included. MNIST variants are downloaded with the setup script `bash setup.sh`.

# Running the experiments

## Weight pruning

`python hibp_weight_pruning.py <args>`

## Permuted MNIST

To run the permuted MNIST experiments

`python run_permuted.py <args>`

## Split MNIST and Variants

`python run_split.py <args>`