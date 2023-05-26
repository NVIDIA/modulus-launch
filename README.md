# Modulus Launch (Beta)

Test PR

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![GitHub](https://img.shields.io/github/license/NVIDIA/modulus-launch)](https://github.com/NVIDIA/modulus-launch/blob/master/LICENSE.txt)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Modulus Launch is a PyTorch based deep-learning collection of training recipes and tools for creating physical surrogates. 
The goal of this repository is to provide a collection of deep learning training examples for different phenomena as starting points for academic and industrial applications.
Additional information can be found in the [Modulus documentation](https://docs.nvidia.com/modulus/index.html#launch).


## Modulus Packages

- [Modulus (Beta)](https://github.com/NVIDIA/modulus)
- [Modulus Launch (Beta)](https://github.com/NVIDIA/modulus-launch)
- [Modulus Symbolic (Beta)](https://github.com/NVIDIA/modulus-sym)
- [Modulus Tool-Chain (Beta)](https://github.com/NVIDIA/modulus-toolchain)

## Installation 

### PyPi

The recommended method for installing the latest version of Modulus Launch is using PyPi:
```Bash
pip install nvidia-modulus.launch
```

### Container

The recommended Modulus docker image can be pulled from the [NVIDIA Container Registry](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/containers/modulus):
```Bash
docker pull nvcr.io/nvidia/modulus/modulus:23.05
```

## From Source

### Package
For a local build of the Modulus Launch Python package from source use:
```Bash
git clone git@github.com:NVIDIA/modulus-launch.git && cd modulus-launch

pip install --upgrade pip
pip install .
```

### Container

To build Modulus Launch docker image:
```
docker build -t modulus-launch:deploy --target deploy -f Dockerfile .
```

To build CI image:
```
docker build -t modulus-launch:ci --target ci -f Dockerfile .
```

To build any of these images on top of the Modulus base image, you can `--build-arg BASE_CONTAINER=modulus:deploy` to the above commands as shown below:
```
docker build --build-arg BASE_CONTAINER=modulus:deploy -t modulus-launch:deploy --target deploy -f Dockerfile .
```

## Contributing

For guidance on making a contribution to Modulus, see the [contributing guidelines](https://github.com/NVIDIA/modulus-launch/blob/main/CONTRIBUTING.md)

## Communication
* Github Discussions: Discuss new architectures, implementations, Physics-ML research, etc. 
* GitHub Issues: Bug reports, feature requests, install issues, etc.
* Modulus Forum: The [Modulus Forum](https://forums.developer.nvidia.com/c/physics-simulation/modulus-physics-ml-model-framework) hosts an audience of new to moderate level users and developers for general chat, online discussions, collaboration, etc. 

## License
Modulus Launch is provided under the Apache License 2.0, please see [LICENSE.txt](./LICENSE.txt) for full license text
