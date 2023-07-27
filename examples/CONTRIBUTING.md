<!-- markdownlint-disable MD043 -->
# Contributing examples to Modulus Repo

## Pre-requisites

1. Read the [code contribution guidelines](https://github.com/NVIDIA/modulus-launch/blob/main/CONTRIBUTING.md)
2. What needs to be part of the submission
    1. Training scripts and corresponding datasets for other community members to replicate
    2. Inference scripts and checkpoints for other community members to replicate
3. Please make sure the code and dataset are cleared to be open-sourced under Apache 2.0
4. Make yourself familiar with Modulus-Core, Modulus-Launch or Modulus-Sym as appropriate:
    1. [Modulus-Core](https://github.com/NVIDIA/modulus) includes a collection of
    models, utilities, and dataloaders
    2. [Modulus-Launch](https://github.com/NVIDIA/modulus-launch) contains a collection
    of training recipes and examples. Modulus is a dependency of Modulus-Launch but not
    vice versa.
    3. [Modulus-Sym](https://github.com/NVIDIA/modulus-sym) provides an abstraction
    layer for using PDE-based symbolic loss functions.

### Recommended Best Practices

1. Wherever possible, avoid  duplicate code. Example:
    1. Use the network implementations already in Modulus.
    2. Try using the dataloaders and utilities that already exists in Modulus.
    3. Trying using the existing logging utilities [example](https://github.com/NVIDIA/modulus-launch/blob/04f598c4556eb598630946816f01dd97467621de/examples/cfd/vortex_shedding_mgn/train.py#L188).
2. Please follow the checkpointing convention/utility in Modulus for saving and loading
your checkpoints [example](https://github.com/NVIDIA/modulus-launch/blob/04f598c4556eb598630946816f01dd97467621de/examples/cfd/vortex_shedding_mgn/train.py#L180).

### Instructions for Contributing Examples

1. Organizing your code and artifacts as follows:
    1. Create a new folder with the same name in the appropriate folders
    2. Architecture/dataloaders/utilities go to Modulus-Core:
        - Network architectures: [under models](https://github.com/NVIDIA/modulus/tree/main/modulus/models)
        - Utilities: [under utils](https://github.com/NVIDIA/modulus/tree/main/modulus/utils)
        - Dataloader: [under datapipes](https://github.com/NVIDIA/modulus/tree/main/modulus/datapipes)
    3. Training/validation/inference scripts and the relevant utilities go to
    Modulus-Launch. Place these scripts in a folder under [examples in Modulus-Launch](https://github.com/NVIDIA/modulus-launch/tree/main/examples).

2. Per the contributing document:
    1. Make forks of Modulus-Core and Modulus-Launch
    2. Push your code to your forks
    3. Once ready, open a PR to the upstream branches
    4. Make sure you prepare your code according to the CI requirements. That includes
    unit tests and code coverage, docstrings, black formatting, and doctests.

3. Please move all of your dependencies to Modulus’s pyproject.toml.

4. Please make sure your dependencies and any of the sub-dependencies are not touching
GPL or L-GPL code. In rare cases, we might be able to clear a code that touches L-GPL
code but there are hard restrictions on GPL code.

5. Please add documentation with all your results and plots as a README.md file in
Modulus_launch. Modulus documentation is being refactored and we will migrate your
documentation to Modulus documentation once ready.
