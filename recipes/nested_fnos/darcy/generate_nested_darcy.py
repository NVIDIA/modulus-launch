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

import hydra
import time
from os.path import isdir
from os import mkdir
from utils import DarcyInset2D
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import numpy as np
import torch


def nested_darcy_generator() -> None:
    """Dataset Generator for the nested Darcy Problem

    This script generates the training, validation and out-of-sample data sets
    for the nested FNO problem and stores them in ./data, where trainer and
    inferencer will find it.
    """
    # out_dir = "./data/"
    # file_names = ["training_data.npy", "validation_data.npy", "out_of_sample.npy"]
    # sample_size = [2048, 256, 128]
    # # sample_size = [8192, 256, 128]
    out_dir = '.'
    file_names = ['test_data.npy']
    sample_size = [64]
    # sample_size = [8192, 256, 128]
    max_batch_size = 128
    resolution = 1024
    glob_res = 256
    fine_res = 128
    buffer = 8
    permea_freq = 5
    fine_permeability_freq = 3
    device = "cuda"
    n_plots = 1

    perm_norm = (0.0, 1.0)
    darc_norm = (0.0, 1.0)

    if not isdir(out_dir):
        mkdir(out_dir)

    assert resolution % glob_res == 0, "resolution needs to be multiple of glob_res"
    ref_fac = resolution // glob_res
    inset_size = fine_res + 2 * buffer
    min_offset = (fine_res * (ref_fac - 1) + 1) // 2 + buffer * ref_fac
    res = np.array([resolution // ref_fac, inset_size])

    # force inset on coarse grid
    if not min_offset % ref_fac == 0:
        min_offset += ref_fac - min_offset % ref_fac

    for dset in range(len(file_names)):
        # compute batch size and number of iterations
        batch_size = min(max_batch_size, sample_size[dset])
        nr_iterations = (sample_size[dset] - 1) // max_batch_size + 1

        datapipe = DarcyInset2D(
            resolution=resolution,
            batch_size=batch_size,
            nr_permeability_freq=permea_freq,
            max_permeability=2.0,
            min_permeability=0.5,
            # max_iterations=30000,
            max_iterations=30000,
            iterations_per_convergence_check=10,
            nr_multigrids=3,
            normaliser={"permeability": perm_norm, "darcy": darc_norm},
            device=device,
            fine_res=fine_res,
            fine_permeability_freq=fine_permeability_freq,
            min_offset=min_offset,
            ref_fac=ref_fac,
        )

        dat = {}
        dat['resolution'] = res
        samp_ind = -1
        for jj, sample in zip(range(nr_iterations), datapipe):
            permea = sample["permeability"].cpu().detach().numpy()
            darcy = sample["darcy"].cpu().detach().numpy()
            pos = (sample["inset_pos"].cpu().detach().numpy()).astype(int)
            assert (pos % ref_fac).sum() == 0, "inset off coarse grid"

            # crop out refined region, allow for surrounding area, save in extra array
            for ii in range(batch_size):
                samp_ind += 1
                samp_str = str(samp_ind)
                dat[samp_str] = {}
                dat[samp_str]['ref0'] = {}
                dat[samp_str]['ref0']['0'] = {}
                dat[samp_str]['ref0']['0']['permeability'] = permea[ii, 0, ::ref_fac, ::ref_fac]
                dat[samp_str]['ref0']['0']['darcy'] = darcy[ii, 0, ::ref_fac, ::ref_fac]

                dat[samp_str]['ref1'] = {}
                for pp in range(pos.shape[1]):
                    xs = pos[ii, pp, 0] - buffer
                    ys = pos[ii, pp, 1] - buffer
                    dat[samp_str]['ref1'][str(pp)] = {}
                    dat[samp_str]['ref1'][str(pp)]['permeability'] = permea[
                        ii, 0, xs : xs + inset_size, ys : ys + inset_size
                    ]
                    dat[samp_str]['ref1'][str(pp)]['darcy'] = darcy[
                        ii, 0, xs : xs + inset_size, ys : ys + inset_size
                    ]
                    dat[samp_str]['ref1'][str(pp)]['pos'] = (pos[ii, pp, :]-min_offset)//ref_fac

        np.save(out_dir + file_names[dset], dat) # TODO track pos min and max and check if within bounds

        # plot coef and solution
        for ii in range(n_plots):
            fields = dat[str(ii)]

            fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5), (ax6, ax7), (ax8, ax9)) = plt.subplots(
                5, 2, figsize=(10, 25)
            )
            ax0.imshow(fields['ref0']['0']['permeability'])
            ax0.set_title("permeability glob")
            ax1.imshow(fields['ref0']['0']['darcy'])
            ax1.set_title("darcy glob")
            ax2.imshow(fields['ref1']['0']['permeability'])
            ax2.set_title("permeability fine 0")
            ax3.imshow(fields['ref1']['0']['darcy'])
            ax3.set_title("darcy fine 0")
            pos = fields['ref1']['0']['pos']
            ax4.imshow(
                fields['ref0']['0']['permeability'][
                    pos[0] : pos[0] + inset_size,
                    pos[1] : pos[1] + inset_size,
                ]
            )
            ax4.set_title("permeability zoomed 0")
            ax5.imshow(
                fields['ref0']['0']['darcy'][
                    pos[0] : pos[0] + inset_size,
                    pos[1] : pos[1] + inset_size,
                ]
            )
            ax5.set_title("darcy zoomed 0")


            ax6.imshow(fields['ref1']['1']['permeability'])
            ax6.set_title("permeability fine 1")
            ax7.imshow(fields['ref1']['1']['darcy'])
            ax7.set_title("darcy fine 1")
            pos = fields['ref1']['1']['pos']
            ax8.imshow(
                fields['ref0']['0']['permeability'][
                    pos[0] : pos[0] + inset_size,
                    pos[1] : pos[1] + inset_size,
                ]
            )
            ax8.set_title("permeability zoomed 1")
            ax9.imshow(
                fields['ref0']['0']['darcy'][
                    pos[0] : pos[0] + inset_size,
                    pos[1] : pos[1] + inset_size,
                ]
            )
            ax9.set_title("darcy zoomed 1")

            fig.tight_layout()
            plt.savefig(f"test_{ii}.png")
            plt.close()


if __name__ == "__main__":
    nested_darcy_generator()
