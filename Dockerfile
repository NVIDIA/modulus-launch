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

ARG BASE_CONTAINER=nvcr.io/nvidia/pytorch:23.06-py3
FROM $BASE_CONTAINER as builder

# Update pip
RUN pip install --upgrade pip 

# CI image
FROM builder as ci
RUN pip install "black==22.10.0" "interrogate==1.5.0" "coverage==6.5.0"
COPY . /modulus-launch/
RUN cd /modulus-launch/ && pip install -e . && rm -rf /modulus-launch/

# Deploy image
FROM builder as deploy
COPY . /modulus-launch/
RUN cd /modulus-launch/ && pip install .

# Clean up
RUN rm -rf /modulus-launch/

# Docs image
FROM deploy as docs
# Install CI packages
RUN pip install "tensorflow>=2.9.0" "warp-lang>=0.6.0" "black==22.10.0" "interrogate==1.5.0" "coverage==6.5.0" "protobuf==3.20.0"
# Install packages for Sphinx build
RUN pip install "recommonmark==0.7.1" "sphinx==5.1.1" "sphinx-rtd-theme==1.0.0" "pydocstyle==6.1.1" "nbsphinx==0.8.9" "nbconvert==6.4.3" "jinja2==3.0.3"
RUN wget https://github.com/jgm/pandoc/releases/download/3.1.6.2/pandoc-3.1.6.2-1-amd64.deb && dpkg -i pandoc-3.1.6.2-1-amd64.deb  
