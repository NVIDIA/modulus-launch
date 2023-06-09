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
    # sample_size = [8192, 256, 128]
    # sample_size = [16384, 256, 128]
    # file_names = ['validation_data.npy']
    # sample_size = [2048]
    out_dir = './'
    file_names = ['out_of_sample_2048.npy']
    sample_size = [2048]
    # sample_size = [8192, 256, 128]
    max_batch_size = 128
    resolution = 1024
    glob_res = 256
    fine_res = 128
    buffer = 8
    permea_freq = 3
    max_n_insets = 3
    fine_permeability_freq = 2
    min_dist_frac = 1.7
    device = "cuda"
    n_plots = 10
    fill_val = -99999

    perm_norm = (0.0, 1.0)
    darc_norm = (0.0, 1.0)

    if not isdir(out_dir):
        mkdir(out_dir)

    assert resolution % glob_res == 0, "resolution needs to be multiple of glob_res"
    ref_fac = resolution // glob_res
    inset_size = fine_res + 2 * buffer
    min_offset = (fine_res * (ref_fac - 1) + 1) // 2 + buffer * ref_fac

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
            max_iterations = 30000,
            iterations_per_convergence_check=10,
            nr_multigrids=3,
            normaliser={"permeability": perm_norm, "darcy": darc_norm},
            device=device,
            max_n_insets=max_n_insets,
            fine_res=fine_res,
            fine_permeability_freq=fine_permeability_freq,
            min_offset=min_offset,
            ref_fac=ref_fac,
            min_dist_frac=min_dist_frac,
            fill_val=fill_val,
        )

        dat = {}
        samp_ind = -1
        for jj, sample in zip(range(nr_iterations), datapipe):
            permea = sample["permeability"].cpu().detach().numpy()
            darcy = sample["darcy"].cpu().detach().numpy()
            pos = (sample["inset_pos"].cpu().detach().numpy()).astype(int)
            assert (np.where(pos==fill_val, 0, pos) % ref_fac).sum() == 0, "inset off coarse grid"

            # crop out refined region, allow for surrounding area, save in extra array
            for ii in range(batch_size):
                samp_ind += 1
                samp_str = str(samp_ind)

                # global fields
                dat[samp_str] = {'ref0': {'0': {'permeability': permea[ii, 0, ::ref_fac, ::ref_fac],
                                                'darcy': darcy[ii, 0, ::ref_fac, ::ref_fac]}}}

                # insets
                dat[samp_str]['ref1'] = {}
                for pp in range(pos.shape[1]):
                    if pos[ii, pp, 0] == fill_val:
                        continue
                    xs = pos[ii, pp, 0] - buffer
                    ys = pos[ii, pp, 1] - buffer

                    dat[samp_str]['ref1'][str(pp)] = \
                            {'permeability': permea[ii, 0, xs : xs + inset_size, ys : ys + inset_size],
                            'darcy': darcy[ii, 0, xs : xs + inset_size, ys : ys + inset_size],
                            'pos': (pos[ii, pp, :]-min_offset)//ref_fac,}

        np.save(out_dir + file_names[dset], dat) # TODO track pos min and max and check if within bounds

        # plot input and target fields
        for jj in range(n_plots):  # TODO move function from evaluate to util and use that one here as well
            fields = dat[str(jj)]
            n_insets = len(fields['ref1'])

            fig, ax = plt.subplots(
                n_insets+1, 4, figsize=(20, 5*(n_insets+1))
            )
            ax[0,0].imshow(fields['ref0']['0']['permeability'])
            ax[0,0].set_title("permeability glob")
            ax[0,1].imshow(fields['ref0']['0']['darcy'])
            ax[0,1].set_title("darcy glob")
            ax[0,2].axis('off')
            ax[0,3].axis('off')

            for ii in range(n_insets):
                loc = fields['ref1'][str(ii)]
                ax[ii+1,0].imshow(loc['permeability'])
                ax[ii+1,0].set_title(f'permeability fine {ii}')
                ax[ii+1,1].imshow(loc['darcy'])
                ax[ii+1,1].set_title(f'darcy fine {ii}')
                ax[ii+1,2].imshow(fields['ref0']['0']['permeability'][
                                    loc['pos'][0] : loc['pos'][0] + inset_size,
                                    loc['pos'][1] : loc['pos'][1] + inset_size,
                                ])
                ax[ii+1,2].set_title(f'permeability zoomed {ii}')
                ax[ii+1,3].imshow(fields['ref0']['0']['darcy'][
                                    loc['pos'][0] : loc['pos'][0] + inset_size,
                                    loc['pos'][1] : loc['pos'][1] + inset_size,
                                ])
                ax[ii+1,3].set_title(f'darcy zoomed {ii}')

            fig.tight_layout()
            plt.savefig(f"sample_{jj:02d}.png")
            plt.close()

if __name__ == "__main__":
    nested_darcy_generator()
