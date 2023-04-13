# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG BASE_CONTAINER=nvcr.io/nvidia/pytorch:22.12-py3
FROM $BASE_CONTAINER as builder

# Update pip
RUN pip install --upgrade pip 

# Install Modulus core
RUN pip install git+https://github.com/NVIDIA/modulus.git@main 

# CI image
FROM builder as ci
RUN pip install black==22.10.0 interrogate==1.5.0 coverage==6.5.0
COPY . /modulus-launch/
RUN cd /modulus-launch/ && pip install -e . && rm -rf /modulus-launch/

# Deploy image
FROM builder as deploy
COPY . /modulus-launch/
RUN cd /modulus-launch/ && pip install .

# Clean up
RUN rm -rf /modulus-launch/ 
