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
This is the code to plot the error of Forward-projection model version , v.s. the VFGN prediction
directly from the /out folders
"""

import numpy as np
from absl import flags,app
import pickle,os,json
import matplotlib.pyplot as plt
# from read_timings_log import get_Newton_iters
import glob, re
import pyvista as pv


def get_solution_id(solution_name):
    m = re.search('displacement-(\d+)', solution_name)
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

    uvw_values = data['displacement_U']

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


err_list = []
err0_list = []
v_list = []
v2_list= []
vel_list = []
acc_list = []
x= []

raw_data_path = "/mnt/archive/becerraj/VFGN/Projects/20220124-PushingGripSolverOnly"
# test builds path
test_raw_data_path_fp = "/mnt/archive/becerraj/VFGN/Projects/20220124-PushingGripInterpolateInitialSolution"
test_raw_data_path_vfgn = "/mnt/archive/becerraj/VFGN/Projects/20220118-PushingGripVFGN"

# read solution files from solver-only
build_path = os.path.join(raw_data_path, 'out')
solution_list = glob.glob(build_path + '/displacement-*.pvtu')
solution_list = sorted(solution_list, key=get_solution_id)
assert len(solution_list) >= 3, "Need to have at least 3 solution files as input to start prediction!"
print("# of solution files: ", len(solution_list))

# read solution files from forward-projection
build_path_fp = os.path.join(test_raw_data_path_fp, 'out')
solution_list_fp = glob.glob(build_path_fp + '/displacement-*.pvtu')
solution_list_fp = sorted(solution_list_fp, key=get_solution_id)
assert len(solution_list_fp) >= 3, "Need to have at least 3 solution files as input to start prediction!"
print("# of solution files fp: ", len(solution_list_fp))

# read solution files from VFGN
build_path_vfgn = os.path.join(test_raw_data_path_vfgn, 'out')
solution_list_vfgn = glob.glob(build_path_vfgn + '/displacement-*.pvtu')
solution_list_vfgn = sorted(solution_list_vfgn, key=get_solution_id)
assert len(solution_list_vfgn) >= 3, "Need to have at least 3 solution files as input to start prediction!"
print("# of solution files vfgn: ", len(solution_list_vfgn))


pos_list, pos_p_list, pos_max_list = [], [], []
pos_list_axis = []
pos_list_fp, pos_p_list_fp, pos_max_list_fp = [], [], []
pos_list_axis_fp = []

pos_list_vfgn, pos_p_list_vfgn, pos_max_list_vfgn = [], [], []
pos_list_axis_vfgn = []

n = len(solution_list)
pid = 50000
sol_index = []
for i in range(0, len(solution_list), 3):
    print("process solution ", os.path.basename(solution_list[i]))
    solution_data = pv.read(solution_list[i])
    pos_array, _ = get_data_position(solution_data)    # displacement
    # print(pos_array.shape)  #(84400, 3)
    pos_list.append(np.mean(pos_array))
    pos_list_axis.append(np.mean(pos_array, axis=0))
    # print(pos_list_axis[0], pos_list_axis[0].shape)
    pos_p_list.append(pos_array[pid])
    pos_max_list.append(np.max(pos_array))

    solution_data_fp = pv.read(solution_list_fp[i])
    pos_array_fp, _ = get_data_position(solution_data_fp)
    pos_list_fp.append(np.mean(pos_array_fp))
    pos_p_list_fp.append(pos_array_fp[pid])
    pos_max_list_fp.append(np.max(pos_array_fp))
    pos_list_axis_fp.append(np.mean(pos_array_fp, axis=0))

    solution_data_vfgn = pv.read(solution_list_vfgn[i])
    pos_array_vfgn, _ = get_data_position(solution_data_vfgn)
    pos_list_vfgn.append(np.mean(pos_array_vfgn))
    pos_p_list_vfgn.append(pos_array_vfgn[pid])
    pos_max_list_vfgn.append(np.max(pos_array_vfgn))
    pos_list_axis_vfgn.append(np.mean(pos_array_vfgn, axis=0))

    sol_index.append(i)


fig, ax = plt.subplots()
# diff_fp = np.absolute(np.array(pos_list_fp)-np.array(pos_list))
# print(diff_fp, diff_fp.shape)

ax.plot(sol_index, pos_list_axis, "b-", linewidth=1, label='solver mean')
ax.plot(sol_index, pos_list_axis_fp, "g-", linewidth=1, label='solver mean abs')
ax.plot(sol_index, pos_list_axis_vfgn, "y-", linewidth=1, label='vfgn')
ax.set_ylabel(" mean deformation (mm)", color="blue", fontsize=14)

fig_name = 'check_consistency_axis'
# ax.set_title('p'+str(pid), fontsize=14)
ax.legend(loc="upper left")
fig.savefig(fig_name+'.jpg',
            format='png',
            dpi=100,
            bbox_inches='tight')
plt.close()

