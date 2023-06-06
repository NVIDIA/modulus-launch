# NVIDIA Modulus Launch Examples

 

## Introduction

This repository provides sample applications demonstrating use of specific Physics-ML model architectures that are easy to train and deploy, to see how such models can help solve real world problems.

## CFD

|Use case|Model|Transient|Parameterized|
| --- | --- |  --- | --- |
|Vortex Shedding|MeshGraphNet|YES|YES|
|Navier-Stokes Flow|RNN|YES|NO|
|Gray-Scott System|RNN|YES|NO|
|Darcy Flow|FNO|NO|YES|
|Darcy Flow|Nested-FNO|NO|YES|
 

## Weather

|Use case|Model|AMP|CUDA Graphs|Multi-GPU| Multi-Node|
| --- | --- | --- | --- | --- | --- |
|Medium-range global weather forecast|FCN-SFNO|YES|NO|YES|YES|
|Medium-range global weather forecast|GraphCast|YES|NO|YES|YES|
|Medium-range global weather forecast|FCN-AFNO|YES|YES|YES|YES|

## NVIDIA support

In each of the network READMEs, we indicate the level of support that will be provided. The range is from ongoing updates and improvements to a point-in-time release for thought leadership.

## Feedback / Contributions

We're posting these examples on GitHub to better support the community, facilitate feedback, as well as collect and implement contributions using GitHub Issues and pull requests. We welcome all contributions!