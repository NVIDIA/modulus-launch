# © Copyright 2023 HP Development Company, L.P.
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


"""
File to check the acceleration profile
"""

import numpy as np
import matplotlib.pyplot as plt
import glob, re, os
import json
import pyvista as pv
from natsort import natsorted
import pickle

from absl import app
from absl import flags


flags.DEFINE_string("raw_data_dir", None, help="Path to rollout pickle file")
# flags.DEFINE_string("rollout_folder_path", None, help="Path to rollout pickle file")
# flags.DEFINE_string("meta_path", None, help="Path to metadata file")

FLAGS = flags.FLAGS


def get_solution_id(solution_name):
    m = re.search('volume-deformation-(\d+)', solution_name)
    if m:
        id = int(m.group(1))
        return id
    return -1


def get_data_position(data):
    """
    For the data read from one displacement-id.pvtu file,
    iterate each point data, filter out the points in existed physical xyz-location
    store the non-repeating point's uvw_values in
    Args:
        data: data read from displacement-id.pvtu file

    Returns: array of non-repeating nodes' current physical location (original location + displacement)

    """
    # Construct a dictionary, store physical location {xyz: boolean}
    arranged_data = {}

    points = data.points
    n_points = points.shape[0]

    uvw_values = data['u__v__w']

    pos_list = []
    receivers_list = []
    index_list=[]

    for point_index in range(n_points):
        # point = data.GetPoint(point_index)
        # point_array=points[point_index]
        # if point not in arranged_data:
        uvw = uvw_values[point_index]

        # Compute the deformed physical location from original physical location
        # pos = point + uvw
        pos = uvw

        index_list.append(point_index)
        pos_list.append(pos)
        # arranged_data[point] = True

    return np.array(pos_list),index_list


def time_diff(sequence_array):
    return sequence_array[1:,:]-sequence_array[:-1,:]


def main(unused_argv):

    build_path = os.path.join(FLAGS.raw_data_dir, 'out')
    # solution_list = glob.glob(build_path + '/displacement-*.pvtu')
    solution_list = glob.glob(build_path + '/volume-deformation-*.pvtu')
    solution_list = sorted(solution_list, key=get_solution_id)
    print("# of solution files vfgn: ", len(solution_list))


    pos_list, pos_p_list, pos_max_list = [], [], []
    pos_list_axis = []
    step = 100

    for i in range(0, len(solution_list), step):
        print("process solution ", os.path.basename(solution_list[i]))
        solution_data = pv.read(solution_list[i])
        pos_array, _ = get_data_position(solution_data)    # displacement
        # print(pos_array.shape)  #(84400, 3)
        # pos_list.append(np.mean(pos_array))
        pos_list.append(pos_array)

        pos_list_axis.append(np.mean(pos_array, axis=0))
        pos_max_list.append(np.max(pos_array))

    print("pos_list: ", np.array(pos_list).shape)
    print("pos_list_axis: ", np.array(pos_list_axis).shape)

    velocity_array = time_diff(np.array(pos_list))
    # velocity_list = [velocity_array[i] for i in range(velocity_array.shape[0])]
    # velocity_list_builds += velocity_list
    vel_3d_mean = np.mean(velocity_array, axis=1)   # mean:  (222880, 3)
    print("vel_3d_mean mean: ", vel_3d_mean.shape)

    acceleration_array = time_diff(velocity_array)
    acceleration_list = [acceleration_array[i] for i in range(acceleration_array.shape[0])]
    # acceleration_list_builds += acceleration_list
    print("acceleration_array: ", acceleration_array.shape)
    print("acceleration_list: ", np.array(acceleration_list).shape)
    print("acceleration_list mean: ", np.mean(acceleration_array, axis=1).shape,
          np.mean(acceleration_array, axis=1))
    acc_3d_mean = np.mean(acceleration_array, axis=1)   # mean:  (222880, 3)
    print("acce_3d_meam ", acc_3d_mean.shape)

    # plot
    build_name = os.path.basename(FLAGS.raw_data_dir)
    fig, ax = plt.subplots()
    sol_index = [i for i in range(acc_3d_mean.shape[0])]
    ax.plot(sol_index, acc_3d_mean[:,0], "b-", linewidth=1, label='x-dim velocity')
    ax.plot(sol_index, acc_3d_mean[:,1], "y-", linewidth=1, label='y-dim velocity')
    ax.plot(sol_index, acc_3d_mean[:,2], "g-", linewidth=1, label='z-dim velocity')
    ax.set_xlabel("time steps", color="blue", fontsize=14)
    ax.set_ylabel("acce", color="blue", fontsize=14)
    # ax.set_ylim(0, 3e-6)

    ax.legend(loc="lower right")
    fig_name = 'acc_3d_'+build_name+'_step'+str(step)
    fig.savefig(fig_name+'.jpg',
                format='png',
                dpi=100,
                bbox_inches='tight')


if __name__ == "__main__":
    app.run(main)