import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_euler_angles
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.animation as animation
matplotlib.use('Agg')  # Use non-interactive backend for video saving

def create_animation(exp_name="live_square_friday", config="quad", fps=30, duration=30):
    # Load data
    data = pd.read_csv("./ulogs/" + exp_name + "/" + exp_name + "_neural_control_0.csv")
    pos_lin_vel_data = pd.read_csv("./ulogs/" + exp_name + "/" + exp_name + "_vehicle_local_position_0.csv")

    # Extract data and remove 0 values from rpm (publishing issue)
    pos_real = np.array([pos_lin_vel_data["x"].tolist(),pos_lin_vel_data["y"].tolist(),pos_lin_vel_data["z"].tolist()])
    pos = np.array([data["observation[0]"].tolist(),data["observation[1]"].tolist(),data["observation[2]"].tolist()])[:,1:]
    lin_vel = np.array([data["observation[9]"].tolist(),data["observation[10]"].tolist(),data["observation[11]"].tolist()])[:,1:]
    ori_6d = np.array([data["observation[3]"].tolist(),data["observation[4]"].tolist(),data["observation[5]"].tolist(),
                       data["observation[6]"].tolist(),data["observation[7]"].tolist(),data["observation[8]"].tolist()])[:,1:]
    ang_vel = np.array([data["observation[12]"].tolist(),data["observation[13]"].tolist(),data["observation[14]"].tolist()])[:,1:]

    if config == "hex":
        m_thrust = np.array([data["motor_thrust[0]"].tolist(),data["motor_thrust[1]"].tolist(),
                            data["motor_thrust[2]"].tolist(),data["motor_thrust[3]"].tolist(),
                            data["motor_thrust[4]"].tolist(),data["motor_thrust[5]"].tolist()])[:,1:]
    elif config == "quad":
        m_thrust = np.array([data["motor_thrust[0]"].tolist(),data["motor_thrust[1]"].tolist(),
                            data["motor_thrust[2]"].tolist(),data["motor_thrust[3]"].tolist()])[:,1:]

    # Transform data
    transformation_1 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    transformation_2 = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pos_enu_real = transformation_1 @ transformation_2 @ pos_real
    ori_matrix = rotation_6d_to_matrix(torch.tensor(ori_6d).T)
    ori_euler = matrix_to_euler_angles(ori_matrix, "XYZ").detach().numpy()

    # Get time points
    uorb_timestep_size = 1/1e6 # one micro second
    start_time = min([data["timestamp"].iloc[0],pos_lin_vel_data["timestamp"].iloc[0]])
    time_points = np.array((data["timestamp"].iloc[1:] - start_time))*uorb_timestep_size
    time_points_pos_vel = np.array((pos_lin_vel_data["timestamp"].iloc[:] - start_time))*uorb_timestep_size

    # Interpolate data
    f_pos_real = interp1d(time_points_pos_vel, pos_enu_real, axis=1)
    f_pos = interp1d(time_points, pos)
    f_lin_vel = interp1d(time_points, lin_vel)
    f_ori = interp1d(time_points, ori_euler.T)
    f_ang_vel = interp1d(time_points, ang_vel, axis=1)
    f_m_thrust = interp1d(time_points, m_thrust)

    sim_dt = 0.01
    T_start = max([time_points[0], time_points_pos_vel[0]])
    T_end = min([time_points[-110], time_points_pos_vel[-1]])
    n_points = int((T_end - T_start)/sim_dt)
    time_points = np.linspace(T_start, T_end, n_points)

    pos_enu_real_interp = f_pos_real(time_points)
    pos_interp = f_pos(time_points)
    lin_vel_interp = f_lin_vel(time_points)
    ori_interp = f_ori(time_points)
    ang_vel_interp = f_ang_vel(time_points)
    m_thrust_interp = f_m_thrust(time_points)*6

    # Create a time array that starts at 0 for plotting purposes
    plotting_time = time_points - T_start + 0.8

    # Setup plot style
    plt.rcParams['text.usetex'] = False
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 12
    plt.style.use("seaborn-v0_8-colorblind")

    # Create figure and axes
    fig, axs = plt.subplots(3, constrained_layout=True, figsize=(10, 8))

    # Set axis limits and labels ahead of time
    for i, ax in enumerate(axs):
        ax.set_xlim(plotting_time[0], plotting_time[-1])

    axs[0].set_ylabel(r"Position Error [$m$]")
    axs[0].set_ylim(-1.6, 1.5)
    axs[0].set_yticks(np.arange(-1.5, 1.6, 0.5))

    axs[1].set_ylabel(r"Velocity [$\frac{m}{s}$]")
    axs[1].set_ylim(-1.8, 1.5)
    axs[1].set_yticks(np.arange(-1.5, 1.6, 0.5))

    axs[2].set_xlabel(r"Time [$s$]")
    axs[2].set_ylabel(r"Motor Thrust [$N$]")
    axs[2].set_ylim(0, 6)

    fig.suptitle("Neural Control in Live Flight", fontsize=20)

    # Initialize empty lines
    pos_lines = [axs[0].plot([], [], label=label)[0]
                for label in [r"$p_x$", r"$p_y$", r"$p_z$"]]
    vel_lines = [axs[1].plot([], [], label=label)[0]
                for label in [r"$v_x$", r"$v_y$", r"$v_z$"]]
    thrust_lines = [axs[2].plot([], [], label=label)[0]
                   for label in [r"$u_1$", r"$u_2$", r"$u_3$", r"$u_4$"]]

    # Add legends
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    # Calculate frames based on duration or default to number of data points
    if duration:
        num_frames = int(fps * duration)
    else:
        num_frames = len(plotting_time)

    # Calculate indices for each frame
    indices = np.linspace(0, len(plotting_time)-1, num_frames, dtype=int)

    def init():
        # Initialize all lines with empty data
        for line in pos_lines + vel_lines + thrust_lines:
            line.set_data([], [])
        return pos_lines + vel_lines + thrust_lines

    def update(frame):
        # Get the index up to which we want to show data
        idx = indices[frame]

        # Update position lines
        for i, line in enumerate(pos_lines):
            line.set_data(plotting_time[:idx+1], pos_interp[i, :idx+1])

        # Update velocity lines
        for i, line in enumerate(vel_lines):
            line.set_data(plotting_time[:idx+1], lin_vel_interp[i, :idx+1])

        # Update thrust lines
        for i, line in enumerate(thrust_lines):
            line.set_data(plotting_time[:idx+1], m_thrust_interp[i, :idx+1])

        return pos_lines + vel_lines + thrust_lines

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, init_func=init, blit=True, interval=500/fps
    )

    # Save animation with lower quality for faster processing
    writer = animation.FFMpegWriter(
        fps=fps,
        metadata=dict(artist='Me'),
        bitrate=300,  # Lower bitrate
        codec='libx264',  # Specify codec directly
        extra_args=['-preset', 'ultrafast', '-crf', '100']  # Faster encoding, lower quality
    )
    ani.save(f"{exp_name}_animated_plot.mp4", writer=writer)

    plt.close()
    print(f"Animation saved as {exp_name}_animated_plot.mp4")

    return ani

if __name__ == "__main__":
    # Create animation with 30 fps, auto-duration (based on data points)
    create_animation(fps=30)

    # Alternatively, specify a fixed duration in seconds:
    # create_animation(fps=30, duration=10)  # 10-second animation