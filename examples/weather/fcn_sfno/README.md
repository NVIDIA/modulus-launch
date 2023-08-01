# Spherical Fourier Neural Operator (SFNO) for weather forecasting

This repository contains the code used for [FourCastNet: A Global Data-driven
High-resolution Weather Model using
Adaptive Fourier Neural Operators](https://arxiv.org/abs/2202.11214)

The code was developed by the authors of the preprint:
Jaideep Pathak, Shashank Subramanian, Peter Harrington, Sanjeev Raja,
Ashesh Chattopadhyay, Morteza Mardani, Thorsten Kurth, David Hall, Zongyi Li,
Kamyar Azizzadenesheli, Pedram Hassanzadeh, Karthik Kashinath, Animashree Anandkumar

## Problem overview

FourCastNet, short for Fourier Forecasting Neural Network, is a global data-driven
weather forecasting model that provides accurate short to medium-range global
predictions at 0.25∘ resolution. FourCastNet accurately forecasts high-resolution,
fast-timescale variables such as the surface wind speed, precipitation, and atmospheric
water vapor. It has important implications for planning wind energy resources,
predicting extreme weather events such as tropical cyclones, extra-tropical cyclones,
and atmospheric rivers. FourCastNet matches the forecasting accuracy of the ECMWF
Integrated Forecasting System (IFS), a state-of-the-art Numerical Weather Prediction
(NWP) model, at short lead times for large-scale variables, while outperforming IFS
for variables with complex fine-scale structure, including precipitation. FourCastNet
generates a week-long forecast in less than 2 seconds, orders of magnitude faster than
IFS. The speed of FourCastNet enables the creation of rapid and inexpensive
large-ensemble forecasts with thousands of ensemble-members for improving probabilistic
forecasting. We discuss how data-driven deep learning models such as FourCastNet are a
valuable addition to the meteorology toolkit to aid and augment NWP models.

FourCastNet is based on the [vision transformer architecture with Adaptive Fourier
Neural Operator (AFNO) attention](https://openreview.net/pdf?id=EXHG-A3jlM)

## Dataset

We rely on DeepMind's vortex shedding dataset for this example. The dataset includes
1000 training, 100 validation, and 100 test samples that are simulated using COMSOL
with irregular triangle 2D meshes, each for 600 time steps with a time step size of
0.01s. These samples vary in the size and the position of the cylinder. Each sample
has a unique mesh due to geometry variations across samples, and the meshes have 1885
nodes on average. Note that the model can handle different meshes with different number
of nodes and edges as the input.

## Model overview and architecture

The model is free-running and auto-regressive. It takes the initial condition as the
input and predicts the solution at the first time step. It then takes the prediction at
the first time step to predict the solution at the next time step. The model continues
to use the prediction at time step $t$ to predict the solution at time step $t+1$, until
the rollout is complete. Note that the model is also able to predict beyond the
simulation time span and extrapolate in time. However, the accuracy of the prediction
might degrade over time and if possible, extrapolation should be avoided unless
the underlying data patterns remain stationary and consistent.

The model uses the input mesh to construct a bi-directional DGL graph for each sample.
The node features include (6 in total):

- Velocity components at time step $t$, i.e., $u_t$, $v_t$
- One-hot encoded node type (interior node, no-slip node, inlet node, outlet node)

The edge features for each sample are time-independent and include (3 in total):

- Relative $x$ and $y$ distance between the two end nodes of an edge
- L2 norm of the relative distance vector

The output of the model is the velocity components at time step t+1, i.e.,
$u_{t+1}$, $v_{t+1}$, as well as the pressure $p_{t+1}$.

![Comparison between the MeshGraphNet prediction and the
ground truth for the horizontal velocity for different test samples.
](../../../docs/img/vortex_shedding.gif)

A hidden dimensionality of 128 is used in the encoder,
processor, and decoder. The encoder and decoder consist of two hidden layers, and
the processor includes 15 message passing layers. Batch size per GPU is set to 1.
Summation aggregation is used in the
processor for message aggregation. A learning rate of 0.0001 is used, decaying
exponentially with a rate of 0.9999991. Training is performed on 8 NVIDIA A100
GPUs, leveraging data parallelism for 25 epochs.

## Getting Started

This example requires the `tensorflow` library to load the data in the `.tfrecord`
format. Install with

```bash
pip install tensorflow
```

To download the data from DeepMind's repo, run

```bash
cd raw_dataset
sh download_dataset.sh cylinder_flow
```

To train the model, run

```bash
python train.py
```

Data parallelism is also supported with multi-GPU runs. To launch a multi-GPU training,
run

```bash
mpirun -np <num_GPUs> python train.py
```

If running in a docker container, you may need to include the `--allow-run-as-root` in
the multi-GPU run command.

Progress and loss logs can be monitored using Weights & Biases. To activate that,
set `wandb_mode` to `online` in the `constants.py`. This requires to have an active
Weights & Biases account. You also need to provide your API key. There are multiple ways
for providing the API key but you can simply export it as an environment variable

```bash
export WANDB_API_KEY=<your_api_key>
```

The URL to the dashboard will be displayed in the terminal after the run is launched.
Alternatively, the logging utility in `train.py` can be switched to MLFlow.

Once the model is trained, run

```bash
python inference.py
```

This will save the predictions for the test dataset in `.gif` format in the `animations`
directory.

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
