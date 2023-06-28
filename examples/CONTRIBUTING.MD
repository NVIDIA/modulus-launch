# Contributing examples to Modulus Repo

## Pre-requisites:
1.Read the contributing document: https://github.com/NVIDIA/modulus/blob/main/CONTRIBUTING.md
2.What needs to be part of the submission
  1.Training scripts and corresponding datasets for other community members to replicate
  2.Inference scripts and checkpoints for other community members to replicate
3.Please make sure the code and dataset are cleared to be open-sourced under Apache 2.0

##General Instructions for adding examples
Make yourself familiar with Modulus-Core, Modulus-Launch or Modulus-Sym as appropriate:
Modulus-Core includes a collection of models, utilities, and dataloaders - https://github.com/NVIDIA/modulus
Modulus-Launch contains a collection of training recipes and examples. Modulus is a dependency of Modulus-Launch but not vice versa - https://github.com/NVIDIA/modulus-launch
Modulus-Sym provides an abstraction layer for using PDE-based symbolic loss functions.  Modulus is a dependency of Modulus-Sym but not vice versa - 
https://github.com/NVIDIA/modulus-sym 

Organizing your code and artifacts as follows:
Generally, model architecture, dataloaders, and utilities go to Modulus-Core. I suggest putting these scripts in Modulus-Core:
Graph_network.py (Ex: network.py) - required if its a new architecture
Utils.py - optional
Reading_utils.py
Dataloader.py
Training/validation/inference scripts and the relevant utilities go to Modulus-Launch.I suggest putting these scripts in Modulus-Launch:
train_global_dynamic_loss.py
train.py
render_rollout.py
Inference_combine_2stages.py
Data_process

Per the contributing document: 
Make forks of Modulus-Core and Modulus-Launch/ Modulus-Sym
Push your code to your forks
Once ready, open an PR to the upstream branches
Make sure you prepare your code according to the CI requirements. That includes unit tests and code coverage, docstrings, black formatting, and doctests.

Please move all of your dependencies to Modulus’s pyproject.toml.

Please make sure your dependencies and any of the sub-dependencies are not touching GPL or L-GPL code. In some cases, we might be able to clear a code that touches L-GPL code but there are hard restrictions on GPL code.

Please add documentation with all your plots and animations as a README.md file in Modulus_launch. Modulus documentation is being refactored and we will migrate your documentation to Modulus documentation once ready.



Recommended Best practices 
Wherever possible, avoid  duplicate code. Example: 
Use the network implementations already in Modulus.
Try using the dataloaders and utilities that already exists in Modulus.
Trying using the existing logging utilities. Example
Please follow the checkpointing convention/utility in Modulus for saving and loading your checkpoints. Example reference. 

