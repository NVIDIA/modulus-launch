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

# ============================================================================
"""Simple matplotlib rendering of a rollout prediction against ground truth.

Usage (from parent directory):

`python -m learning_to_simulate.render_rollout --rollout_path={OUTPUT_PATH}/rollout_test_1.pkl`

Where {OUTPUT_PATH} is the output path passed to `train.py` in "eval_rollout"
mode.

It may require installing Tkinter with `sudo apt-get install python3.7-tk`.

"""

import pickle
import os, json
import numpy as np

from absl import app
from absl import flags

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation


flags.DEFINE_string("rollout_path", "rollouts/rollout_test_0.pkl", help="Path to rollout pickle file")
flags.DEFINE_string("metadata_path", "data", help="Path to metadata file")
flags.DEFINE_integer("step_stride", 3, help="Stride of steps to skip.")
flags.DEFINE_boolean("block_on_show", True, help="For test purposes.")
flags.DEFINE_enum('ds_type', 'standard', ['standard', 'train', 'test'],
                  help='test data type.')

flags.DEFINE_string("test_build", "test0", help="Test build name")
flags.DEFINE_boolean("plot_tolerance_range", True, help="For test purposes.")
flags.DEFINE_boolean("plot_3d", False, help="For test purposes.")

FLAGS = flags.FLAGS


def compute_accuracy_percent(rollout_data, trajectory_len):
    """
    This function compute the percentage accuracy of uvw-channels:
        (abs(gt- pred)) / gt_dispalcement

    :param rollout_data:
    :param trajectory_len: num of rollout steps
    :return:
        percentage_rollout_list: mean accuracy (%) of each timestep, across 3-channels
        percent_uvw: (dim1: timesteps, dim2: 3-dimensions)
    """
    init_position = rollout_data["initial_positions"][0, ...]
    gt_position = rollout_data['ground_truth_rollout']
    pred_position = rollout_data['predicted_rollout']

    percentage_rollout_list = []
    percent_uvw = []
    for i in range(trajectory_len):
        # Iterate for each rollout trajectory step, each step shape, i.e.:(21969, 3)
        # Compute the different percentage: (abs(gt- pred)) / gt_dispalcement

        # Compute the mean diff of 3-channels
        diff = np.float64((gt_position[i, ...] - pred_position[i, ...]) ** 2)
        diff_point = np.sqrt(np.sum(diff, axis=1))

        gt_displacement = np.float64((gt_position[i, ...] - init_position) ** 2)
        gt_displacement_point = np.sqrt(np.sum(gt_displacement, axis=1))

        # Set the episilon (mean diff_point: e-05, mean gt_displacement_point: e-03)
        e = 1e-07
        # Exclude the point where displacement == 0
        nonzero_index = np.where(gt_displacement_point != 0)[0]
        zero_index = np.where(gt_displacement_point == 0)[0]
        # percent_point = np.mean(diff_point[nonzero_index] / gt_displacement_point[nonzero_index])
        percent_point_2 = np.mean(np.mean(diff_point[nonzero_index] / gt_displacement_point[nonzero_index]) + \
                                  np.mean(diff_point[zero_index] / e))

        percent_point = np.sum(diff_point) / (np.sum(gt_displacement_point) + e)
        print(f"mean diff_point: {np.sum(diff_point)}, mean gt_displacement_point: {np.sum(gt_displacement_point)}")
        print(percent_point, percent_point_2)

        # Compute mean for 3 axis, represent u, v, w values
        diff_abs = np.absolute(np.float64((gt_position[i, ...] - pred_position[i, ...])))
        gt_displacement_abs = np.absolute(np.float64((gt_position[i, ...] - init_position)))
        percent_uvw_time = np.sum(diff_abs, axis=0) / np.sum(gt_displacement_abs, axis=0)

        percentage_rollout_list.append(percent_point)
        percent_uvw.append(percent_uvw_time)
    print(percentage_rollout_list)
    return percentage_rollout_list, np.array(percent_uvw)


def plot_rollout_percentage(percentage_rollout_list, percent_uvw,
                            save_path, save_name, build_name="ladder"):

    """
    bar plot of rollout percentage loss
    Args:
        percentage_rollout_list: ave of 3-dims percentage loss
        percent_uvw: (trajectory_length, dim=3)
    Returns:
        None
    """
    print("plot_rollout_percentage, num of rollout steps: ", len(percentage_rollout_list))
    n = len(percentage_rollout_list)
    fig = plt.figure(figsize=(20, 5))

    # creating the bar plot, plot the mean accuracy across uvw 3-channels
    # x-axis: rollout timsteps, y-axis: accuracy
    name_list = [str(x) for x in range(len(percentage_rollout_list))]
    name_values = [x for x in range(len(percentage_rollout_list))]
    plt.bar(name_list, percentage_rollout_list, color="silver")

    # Add u, v, w accuracy curves on the plot
    plt.plot(name_values, percent_uvw[:, 0], "b-", label='u-displacement')
    plt.plot(name_values, percent_uvw[:, 1], "g-", label='v-displacement')
    plt.plot(name_values, percent_uvw[:, 2], "y-", label='w-displacement')
    plt.legend(["u-displacement", "v-displacement", "w-displacement"], loc="lower right")
    plt.legend()

    # Add 10% cut-off line, this is the accuracy tolerance requirement
    cutoff_line = [0.1 for i in range(len(percentage_rollout_list))]
    plt.plot(name_values, cutoff_line, "r-")

    cutoff_line = [0.03 for i in range(len(percentage_rollout_list))]
    plt.plot(name_values, cutoff_line, "r.")

    # Set x-y axis range
    plt.xlim(0, n)
    plt.ylim(0, 0.2)

    plt.xlabel("Rollout steps")
    plt.ylabel("(abs(gt- pred)) / gt (%)")
    plt.title("Percent loss as compare to VF  " + build_name)
    plt.savefig(os.path.join(save_path, save_name + "_" + build_name + ".png"))
    plt.close()


def plot_3Danime(rollout_data, pred_denorm, save_name):
    print("\n\nplot_3Danime: ")
    fig= plt.figure(figsize=(10, 5))
    plot_info = []
    # choose the bounds set in the metadata, or manually set plot bounds
    bounds = rollout_data["metadata"]["bounds"]
    bounds = [[-1.5, 1.5], [-1.5, 1.5], [-1, 0]]

    for ax_i, (label, rollout_field) in enumerate(
            [("Ground truth", "ground_truth_rollout"),
             ("Prediction", "predicted_rollout")]):
        # Append the initial positions to get the full trajectory.
        ax = fig.add_subplot(1, 2, (ax_i+1), projection='3d')
        # title = label
        title = ax.set_title(label)
        ax.set_xlim3d(bounds[0][0], bounds[0][1])
        # ax.set_xticks(np.arange(bounds[0][0]-0.25, bounds[0][1]+0.25, 0.5))
        ax.set_xlabel('X')
        ax.set_ylim3d(bounds[1][0], bounds[1][1])
        # ax.set_yticks(np.arange(bounds[1][0]-0.25, bounds[1][1]+0.25, 0.5))
        ax.set_ylabel('Y')
        ax.set_zlim3d(bounds[2][0], bounds[2][1])
        ax.set_zlabel('Z')
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        # ax.view_init(40, 50)
        ax.auto_scale_xyz
        ax.view_init(40, 55)

        data = rollout_data[rollout_field][0, ...]
        graph, = ax.plot(data[:, 0], data[:, 1], data[:, 2],
                         linestyle="", marker="o", ms=1)
        points = rollout_data[rollout_field]
        #         points = {
        #             particle_type: ax.scatter3D([], [], [], "o", color=color)[0]
        #             for particle_type, color in TYPE_TO_COLOR.items()}
        plot_info.append((ax, label, points, graph))

    num_steps = pred_denorm.shape[0]
    print("predicted shape: ", num_steps, pred_denorm.shape)
    def update_graph(num):
        outputs = []
        for _, label, points, graph in plot_info:
            # todo: append mask to points info
            data = points[num, ...]
            graph.set_data(data[:, 0], data[:, 1])
            graph.set_3d_properties(data[:, 2])
            title.set_text('{}, time={}'.format(label, num))
            outputs.append(graph)
        return outputs
        # return title, graph,

    ani = animation.FuncAnimation(fig, update_graph,
                                  num_steps,
                                  interval=70, blit=False, repeat=True)

    # Save gif
    save = True
    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
        ani.save(save_name+'-3d-animated.mp4', writer=writer)

    plt.close()


def plot_mean_error(rollout_data, metadata, plot_steps, build_name):
    gt_position = rollout_data['ground_truth_rollout']
    pred_position = rollout_data['predicted_rollout']
    # print("rollout shape: ", pred_position.shape) rollout shape:  (164, 100764, 3)

    pos_mean = metadata['pos_mean']
    pos_std = metadata['pos_std']

    gt_position = gt_position * pos_std + pos_mean
    pred_position = pred_position * pos_std + pos_mean

    rollout_list = []
    rollout_list_max = []
    rollout_uvw = []
    for i in range(plot_steps):
        # Compute the mean diff of 3-channels
        diff = np.absolute(gt_position[i, ...] - pred_position[i, ...])
        me_ = np.mean(diff)
        max_ = np.max(diff)
        print("step me, max: ", i, me_, max_)

        # Compute mean for 3 axis, represent u, v, w values
        uvw_time = np.mean(diff, axis=0)

        rollout_list.append(me_)
        rollout_list_max.append(max_)
        rollout_uvw.append(uvw_time)

    print("rolloutlist shape :", np.array(rollout_list).shape)
    print("rollout_uvw shape :", np.array(rollout_uvw).shape)

    ########## Plot ##########
    fig = plt.figure(figsize=(20, 9))
    # x-axis: rollout timsteps, y-axis: accuracy
    name_list = [str(x) for x in range(len(rollout_list))]
    name_values = [x for x in range(len(rollout_list))]
    # plt.bar(name_list, rollout_list, color="silver")
    plt.bar(name_list, rollout_list, color="silver")
    plt.plot(name_values, rollout_list_max, "r.")

    # Add u, v, w accuracy curves on the plot
    ######### Comment out for adding the xyz-deformation curves
    # rollout_uvw = np.array(rollout_uvw)
    # plt.plot(name_values, rollout_uvw[:, 0], "b-", label='u-displacement')
    # plt.plot(name_values, rollout_uvw[:, 1], "g-", label='v-displacement')
    # plt.plot(name_values, rollout_uvw[:, 2], "y-", label='w-displacement')
    # plt.legend(["u-displacement", "v-displacement", "w-displacement"],
    #            loc="lower right", prop={'size': 10})
    # plt.legend()

    # cutoff_line = [0.001 for i in range(len(name_values))]
    # plt.plot(name_values, cutoff_line, "r.")

    # Set x-y axis range
    plt.xlim(0, plot_steps)
    # plt.ylim(0, 0.05)
    plt.xticks(np.arange(0, plot_steps, 10))
    # plt.yticks(np.arange(0, 250, 50))
    plt.xticks(size=30)
    plt.yticks(size=30)

    plt.xlabel("Rollout steps", fontsize=30)
    plt.ylabel("Accuracy (Mean error/mm)", fontsize=30)
    # plt.title("Mean error as compare to VF  " + build_name)
    plt.savefig(os.path.join(os.path.dirname(FLAGS.rollout_path), "mean_error_" + build_name + ".png"))
    plt.close()

    return rollout_list, rollout_uvw


def plot_mean_error_temperature(rollout_data, metadata, plot_steps, build_name):
    gt_position = rollout_data['ground_truth_rollout']
    pred_position = rollout_data['predicted_rollout']
    temperatures = rollout_data['global_context'][3:]
    print("rollout shape: ", pred_position.shape) # rollout shape:  (164, 100764, 3)
    print("temperature: ", temperatures.shape)

    pos_mean = metadata['pos_mean']
    pos_std = metadata['pos_std']

    gt_position = gt_position * pos_std + pos_mean
    pred_position = pred_position * pos_std + pos_mean

    rollout_list = []
    rollout_uvw = []
    for i in range(plot_steps):
        # Compute the mean diff of 3-channels
        diff = np.absolute(gt_position[i, ...] - pred_position[i, ...])
        me_ = np.mean(diff)
        # print("step me: ", i, me_)

        # Compute mean for 3 axis, represent u, v, w values
        uvw_time = np.mean(diff, axis=0)

        rollout_list.append(me_*1000)
        rollout_uvw.append(uvw_time*1000)

    print("rolloutlist shape :", np.array(rollout_list).shape)
    print("rollout_uvw shape :", np.array(rollout_uvw).shape)

    ########## Plot ##########
    fig, ax = plt.subplots(figsize=(20, 7))
    name_values = [x for x in range(len(rollout_list))]

    # Add u, v, w accuracy curves on the plot
    rollout_uvw = np.array(rollout_uvw)
    ax.plot(name_values, rollout_uvw[:, 0], "b-", label='u-displacement')
    ax.plot(name_values, rollout_uvw[:, 1], "g-", label='v-displacement')
    ax.plot(name_values, rollout_uvw[:, 2], "y-", label='w-displacement')
    ax.legend(["u-displacement", "v-displacement", "w-displacement"],
              loc="lower right", prop={'size': 10})
    ax.legend()

    ax.set_xlabel("Rollout steps", fontsize=20)
    ax.set_ylabel("Accuracy (Mean error/mm)", fontsize=20)

    # twin object for two different y-axis on the sample plot
    ax2 = ax.twinx()
    # make a plot with different y-axis using second axis object
    temperatures_list = temperatures.flatten()
    name_values = [x for x in range(len(temperatures_list))]
    # print("temperature: ", temperatures_list, temperatures_list.shape)
    ax2.plot(name_values, temperatures_list, "r-", linewidth=2)
    ax2.set_ylabel("Temperature", color="red", fontsize=14)

    plt.title("Mean error as compare to VF  " + build_name)
    plt.savefig(os.path.join(os.path.dirname(FLAGS.rollout_path), "mean_error_wtemp" + build_name + ".png"))
    plt.close()

    return rollout_list, rollout_uvw


def main(unused_argv):
    if not FLAGS.rollout_path:
        raise ValueError("A `rollout_path` must be passed.")
    with open(FLAGS.rollout_path, "rb") as file:
        rollout_data = pickle.load(file)

    with open(os.path.join(FLAGS.metadata_path, "metadata.json"), 'r') as f:
        metadata = json.load(f)

    pos_mean = metadata['pos_mean']
    pos_std = metadata['pos_std']
    print("mean, std: ", pos_mean, pos_std)

    seq_len = metadata['sequence_length']
    seq_len = 54
    n = seq_len-len(rollout_data["initial_positions"]) -1
    print("initial steps #=", len(rollout_data["initial_positions"]))
    print("pred steps #=", len(rollout_data["predicted_rollout"]))
    print("n: ", n)

    n = len(rollout_data["predicted_rollout"])

    for i in range(n):
        A = rollout_data['ground_truth_rollout'][i]
        A = pos_std*A+pos_mean

        B = rollout_data['predicted_rollout'][i]
        B = pos_std*B+pos_mean

        C = A-B
        C = C.reshape((-1,3))
        mse0 = np.square(C)
        me0 = np.absolute(C)

        mse=np.mean(mse0)
        me=np.mean(me0)
        # print(f"ground truth: {A}, me: {me}")

        print(f"{i} step, me: {me}, mse: {mse}")

    A = rollout_data['ground_truth_rollout'][:n, ...]
    A = pos_std * A + pos_mean

    B = rollout_data['predicted_rollout'][:n, ...]
    B = pos_std * B + pos_mean
    C = A - B
    C = C.reshape((-1, 3))
    mse0 = np.square(C)
    me0 = np.absolute(C)

    mse = np.mean(mse0)
    me = np.mean(me0)
    print(f"rollout shape: {A.shape}, total me: {me}")


    ############ PLOT ############
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # If plot tolerance range
    print("Compute percentage rollout \n\n")
    if FLAGS.plot_tolerance_range:
        percentage_rollout_list, percent_uvw = compute_accuracy_percent(rollout_data, n)
        plot_rollout_percentage(percentage_rollout_list, percent_uvw,
                                os.path.dirname(FLAGS.rollout_path), "rollout_acc_percent",
                                build_name=FLAGS.test_build)

    print("\n\n plot mean error")
    # plot_mean_error_temperature(rollout_data, metadata, n, FLAGS.test_build)
    plot_mean_error(rollout_data, metadata, n, FLAGS.test_build)

    if FLAGS.plot_3d:
        # Plot 3D visualization
        # gt_denorm = (rollout_data['ground_truth_rollout']) * pos_std + pos_mean
        pred_denorm = (rollout_data['predicted_rollout']) * pos_std + pos_mean
        plot_3Danime(rollout_data, pred_denorm, FLAGS.rollout_path[:-4])


if __name__ == "__main__":
    app.run(main)
