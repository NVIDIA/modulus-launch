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

import torch
import torch.nn as nn

from functools import partial

from modulus.models.sfno.preprocessor import Preprocessor2D
from modulus.models.sfno.sfnonet import SphericalFourierNeuralOperatorNet


class SingleStepWrapper(nn.Module):
    """Wrapper for training a single step into the future"""

    def __init__(self, params, model_handle):
        super(SingleStepWrapper, self).__init__()
        self.preprocessor = Preprocessor2D(params)
        self.model = model_handle(params)

    def forward(self, inp):
        # first append unpredicted features
        inpa = self.preprocessor.append_unpredicted_features(inp)

        # now normalize
        self.preprocessor.history_compute_stats(inpa)
        inpan = self.preprocessor.history_normalize(inpa, target=False)

        # now add static features if requested
        inpans = self.preprocessor.add_static_features(inpan)

        # forward pass
        yn = self.model(inpans)

        # undo normalization
        y = self.preprocessor.history_denormalize(yn, target=True)

        return y


class MultiStepWrapper(nn.Module):
    """Wrapper for training multiple steps into the future"""

    def __init__(self, params, model_handle):
        super(MultiStepWrapper, self).__init__()
        self.preprocessor = Preprocessor2D(params)
        self.model = model_handle(params)

        # collect parameters for history
        self.n_future = params.n_future

    def _forward_train(self, inp):

        result = []
        inpt = inp
        for step in range(self.n_future + 1):

            # add unpredicted features
            inpa = self.preprocessor.append_unpredicted_features(inpt)

            # do history normalization
            self.preprocessor.history_compute_stats(inpa)
            inpan = self.preprocessor.history_normalize(inpa, target=False)

            # add static features
            inpans = self.preprocessor.add_static_features(inpan)

            # prediction
            predn = self.model(inpans)

            # append the denormalized result to output list
            # important to do that here, otherwise normalization stats
            # will have been updated later:
            pred = self.preprocessor.history_denormalize(predn, target=True)
            result.append(pred)

            if step == self.n_future:
                break

            # append history
            inpt = self.preprocessor.append_history(inpt, pred, step)

        # concat the tensors along channel dim to be compatible with flattened target
        result = torch.cat(result, dim=1)

        return result

    def _forward_eval(self, inp):
        # first append unpredicted features
        inpa = self.preprocessor.append_unpredicted_features(inp)

        # do history normalization
        self.preprocessor.history_compute_stats(inpa)
        inpan = self.preprocessor.history_normalize(inpa, target=False)

        # add static features
        inpans = self.preprocessor.add_static_features(inpan)

        # important, remove normalization here,
        # because otherwise normalization stats are already outdated
        yn = self.model(inpans)

        # important, remove normalization here,
        # because otherwise normalization stats are already outdated
        y = self.preprocessor.history_denormalize(yn, target=True)

        return y

    def forward(self, inp):
        # decide which routine to call
        if self.training:
            y = self._forward_train(inp)
        else:
            y = self._forward_eval(inp)

        return y


def get_model(params):
    model_handle = None

    if params.nettype == "sfno":
        assert params.spectral_transform == "sht"

        # use the Helmholtz decomposition

        model_handle = partial(
            SphericalFourierNeuralOperatorNet, use_complex_kernels=True
        )
    else:
        raise NotImplementedError(f"Error, net type {params.nettype} not implemented")

    # wrap into Multi-Step if requested
    if params.n_future > 0:
        model = MultiStepWrapper(params, model_handle)
    else:
        model = SingleStepWrapper(params, model_handle)

    return model
