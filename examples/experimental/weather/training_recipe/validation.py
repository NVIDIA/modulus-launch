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
import matplotlib.pyplot as plt

from data_helpers import concat_static_features


@torch.inference_mode()
def validation_step(eval_step, model, datapipe, channels=[0, 1], epoch=0):
    loss_epoch = 0
    num_examples = 0  # Number of validation examples
    # Dealing with DDP wrapper
    if hasattr(model, "module"):
        model = model.module
    model.eval()
    for i, data in enumerate(datapipe):
        data = data[0]
        invar = data["state_seq"][:, 0]
        invar = concat_static_features(invar, data, step=0).detach()
        outvar = data["state_seq"][:, 1:].cpu().detach()
        predvar = torch.zeros_like(outvar)

        for t in range(outvar.shape[1]):
            output = eval_step(model, invar)
            invar[:, 0 : output.size(1)].copy_(output)
            invar = concat_static_features(
                invar,
                data,
                step=t + 1,
                update_coszen=True,
                coszen_channel=output.size(1),
            ).detach()
            predvar[:, t] = output.detach().cpu()

        num_elements = torch.prod(torch.Tensor(list(predvar.shape[1:])))
        loss_epoch += torch.sum(torch.pow(predvar - outvar, 2)) / num_elements
        num_examples += predvar.shape[0]

        # Plotting
        if i == 0:
            predvar = predvar.numpy()
            outvar = outvar.numpy()
            for chan in channels:
                plt.close("all")
                fig, ax = plt.subplots(
                    3, predvar.shape[1], figsize=(15, predvar.shape[0] * 5)
                )
                for t in range(outvar.shape[1]):
                    ax[0, t].imshow(predvar[0, t, chan])
                    ax[1, t].imshow(outvar[0, t, chan])
                    ax[2, t].imshow(predvar[0, t, chan] - outvar[0, t, chan])

                fig.savefig(f"era5_validation_channel{chan}_epoch{epoch}.png")

    model.train()
    return loss_epoch / num_examples
