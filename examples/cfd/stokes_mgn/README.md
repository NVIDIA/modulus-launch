# Learning the flow field of Stokes flow 

This example demonstrates how to train the MeshGraphNet model to learn the flow field of Stokes flow and further 
finetune the model predictions by physics-informed learning. 


## Problem overview

The partial differential equation is defined as 

$$\begin{aligned}
    -\nu \Delta \mathbf{u} +\nabla p=0, \\
    \nabla \cdot \mathbf{u} = 0,
\end{aligned}$$

where $\mathbf{u} = (u, v)$ defines the velocity and $p$ the pressure, and $\nu$ is the kinematic viscosity. 
The underlying geometry is a pipe without a polygon. On the inlet $\Gamma_3=0 \times[0,0.41]$, a parabolic inflow profile is prescribed,

$$\begin{aligned}
    \mathbf{u}(0, y)= \mathbf{u}_{\mathrm{in}} = \left(\frac{4 U y(0.41-y)}{0.41^2}, 0\right)
\end{aligned}$$

with a maximum velocity $U=0.3$. On the outlet $\Gamma_4=2.2 \times[0,0.41]$, we define the outflow condition

$$\begin{aligned}
    \nu \partial_\mathbf{n} \mathbf{u}-p \mathbf{n}=0,
\end{aligned}$$

where $\mathbf{n}$ denotes the outer normal vector.

Our goal is to train the map from the polygon geometry to the velocity and pressure field. 

## Dataset

Our dataset describes fluid flow in a pipe domain obstructed by a random polygon. 
It contains information about the flow velocity, pressure, and markers identifying different boundaries within the domain.
It contains 1000 random samples and all the simulations were performed  using Fenics.

To request access to the full dataset, please reach out to the
[NVIDIA Modulus team](simnet-team@nvidia.com).

## Model overview and architecture

Our model is MeshGraphNet.  The inputs to the model are:

- mesh

Output of the model are:

- velocity field pressure
- pressure field

The input to the model is in form of a `.vtp` file and is then converted to
bi-directional DGL graphs in the dataloader. The final results are also written in the
form of `.vtp` files in the inference code. A hidden dimensionality of 256 is used in
the encoder, processor, and decoder. The encoder and decoder consist of two hidden
layers, and the processor includes 15 message passing layers. Batch size per GPU is
set to 1. Summation aggregation is used in the
processor for message aggregation. A learning rate of 0.0001 is used, decaying
exponentially with a rate of 0.99985.

![Comparison between the AeroGraphNet prediction and the
ground truth for surface pressure, wall shear stresses, and the drag coefficient for one
of the samples from the test dataset.](../../../docs/img/ahmed_body_results.png)


## Getting Started

The dataset for this example is not publicly available. To get access, please reach out
to the [NVIDIA Modulus team](simnet-team@nvidia.com).

This example requires the `pyvista` and `vtk` libraries. Install with

```bash
pip install pyvista vtk
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

To further fine-tune the model using physics-informed learning, run

```bash
python pi_inference.py
```

This will save the predictions for the test dataset in `.vtp` format in the `results`
directory. Use Paraview to open and explore the results.

## References

- [Learning Mesh-Based Simulation with Graph Networks](https://arxiv.org/abs/2010.03409)
