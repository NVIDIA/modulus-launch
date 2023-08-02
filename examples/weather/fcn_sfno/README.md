# Spherical Fourier Neural Operator (SFNO) for weather forecasting

This repository contains the code used for [Spherical Fourier Neural Operators:
Learning Stable Dynamics on the Sphere](https://arxiv.org/abs/2306.03838)

The code was developed by the authors of the preprint:
Boris Bonev, Thorsten Kurth, Christian Hundt, Jaideep Pathak, Maximilian Baust,
Karthik Kashinath, Anima Anandkumar. Other contributors are David Pruitt, Jean Kossaifi,
and Noah Brenowitz.

## Problem overview

Fourier Neural Operators (FNOs) have proven to be an efficient and effective method for
resolution independent operator learning in a broad variety of application areas across
scientific machine learning. A key reason for their success is their
ability to accurately model long-range dependencies in spatio-temporal data by learning
global convolutions in a computationally efficient manner. To this end, FNOs rely on the
discrete Fourier transform (DFT), however, DFTs cause visual and spectral artifacts as
well as pronounced dissipation when learning operators in spherical coordinates since
they incorrectly assume a flat geometry. To overcome this limitation, FNOs are
generalized on the sphere by introducing Spherical FNOs (SFNOs) for learning
operators on spherical
geometries. In this example, SFNO is applied to forecasting atmospheric dynamics,
and demonstrates stable autoregressive rollouts for a year of simulated time
(1,460 steps), while retaining
physically plausible dynamics. The SFNO has important implications for machine
learning-based simulation of climate dynamics that could eventually help accelerate
our response to climate change.

## Dataset

The model is trained on a subset of the ERA5 reanalysis data on single levels and
pressure levels that is pre-processed and stored into HDF5 files.
A small subset of the ERA5 training data is hosted at the
National Energy Research Scientific Computing Center (NERSC). For convenience
[it is available to all via Globus](https://app.globus.org/file-manager?origin_id=945b3c9e-0f8c-11ed-8daf-9f359c660fbd&origin_path=%2F~%2Fdata%2F).
You will need a Globus account and will need to be logged in to your account in order
to access the data.  You may also need the [Globus Connect](https://www.globus.org/globus-connect)
to transfer data.

## Model overview and architecture

Please refer to the [reference paper](https://arxiv.org/abs/2306.03838) to learn about
the model architecture.

## Getting Started

To train the model on a single GPU, run

```bash
python train.py --yaml_config=config/sfnonet.yaml --config=base_config
```

This will launch a SFNO training using the base configs specified in the
`config/sfnonet.yaml` file.

You can include other arguments in the run command to change some of the defaults.
For example, for a mock-up training using synthetic data, run

```bash
python train.py --yaml_config=config/sfnonet.yaml --config=base_config --enable_synthetic_data
```

Other arguments include, but not limited to:

- fin_parallel_size: Input feature parallelization
- fout_parallel_size: Output feature parallelization
- h_parallel_size: Spatial parallelism dimension in h
- w_parallel_size: Spatial parallelism dimension in w
- amp mode: The mixed precision mode
- cuda_graph_mode: Specified which parts of the training to capture under CUDA graphs
- checkpointing_level: Specifies how aggressively the gradient checkpointing is used
- mode: Specifies the run mode, i.e., training, inference, or ensemble.
- multistep_count: Number of autoregressive training steps.

To see a list of all available options with description, run

```bash
python train.py --help
```

Other configurations that are not covered by these options can be specified or
overwritten in the config files. Those include, but not limited to, the loss function
type, learning rate, path to the dataset, number of channels, type of activation or
normalization, number of layers, etc.

Progress and loss logs can be monitored using Weights & Biases. This requires to have an
active Weights & Biases account. You also need to provide your API key. There are
multiple ways for providing the API key but you can simply export it as an environment
variable

```bash
export WANDB_API_KEY=<your_api_key>
```

The URL to the dashboard will be displayed in the terminal after the run is launched.

If needed, Weights & Biases can be disabled by

```bash
export WANDB_MODE='disabled'
```

## References

If you find this work useful, cite it using:

```text
@article{bonev2023spherical,
  title={Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere},
  author={Bonev, Boris and Kurth, Thorsten and Hundt, Christian and Pathak, Jaideep
          and Baust, Maximilian and Kashinath, Karthik and Anandkumar, Anima},
  journal={arXiv preprint arXiv:2306.03838},
  year={2023}
}
```

ERA5 data was downloaded from the Copernicus Climate Change Service (C3S)
Climate Data Store.

```text
Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J.,
Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., 
Dee, D., Thépaut, J-N. (2018): ERA5 hourly data on pressure levels from 1959 to present.
Copernicus Climate Change Service (C3S) Climate Data Store (CDS). 10.24381/cds.bd0915c6

Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J.,
Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C.,
Dee, D., Thépaut, J-N. (2018): ERA5 hourly data on single levels from 1959 to present.
Copernicus Climate Change Service (C3S) Climate Data Store (CDS). 10.24381/cds.adbb2d47
```
