import sys

# Add the correct path to system path
sys.path.insert(0, '/home/itk/Desktop/dev_sindre/aerial_gym_simulator_sindre')

import time
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.registry.task_registry import task_registry
import torch

from aerial_gym.examples.rl_games_example.rl_games_inference import MLP

import time
import numpy as np

import matplotlib.pyplot as plt

if __name__ == "__main__":
    logger.print_example_message()
    start = time.time()
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    plt.style.use("seaborn-v0_8-colorblind")
    rl_task_env = task_registry.make_task(
        # "position_setpoint_task_sim2real_end_to_end",
        "position_setpoint_task_sim2real_px4",
        # other params are not set here and default values from the task config file are used
        seed=seed,
        headless=True,
        num_envs=24,
        use_warp=True,
    )
    rl_task_env.reset()
    actions = torch.zeros(
        (
            rl_task_env.sim_env.num_envs,
            rl_task_env.task_config.action_space_dim,
        )
    ).to("cuda:0")
    model = (
        MLP(
            rl_task_env.task_config.observation_space_dim,
            rl_task_env.task_config.action_space_dim,
            # "networks/morphy_policy_for_rigid_airframe.pth"
            # "networks/attitude_policy.pth"
            "/home/itk/Desktop/dev_sindre/aerial_gym_simulator_sindre/resources/conversion/gen_ppo.pth"
            # "networks/morphy_policy_for_flexible_airframe_joint_aware.pth",
        )
        .to("cuda:0")
        .eval()
    )
    actions[:] = 0.0
    counter = 0
    action_list = []
    obs_list = []
    euler_angle_list = []
    robot_position_list = []

    rl_task_env.task_config.episode_len_steps = 100000 # Don't reset
    rl_task_env.target_position[:] = torch.tensor([0.0, 0.0, 1.0]).to("cuda:0")

    obs_dict = rl_task_env.obs_dict
    print("Starting simulation")
    with torch.no_grad():
        for i in range(3200): #8600
            if i == 700:
                print("Changing target position to [-0.7, 0.7, 0.0]")
                rl_task_env.target_position[:] = torch.tensor([0.7, 0.7, 1.0]).to("cuda:0")
            if i == 1200:
                print("Changing target position to [-0.7, -0.7, 0.0]")
                rl_task_env.target_position[:] = torch.tensor([-0.7, 0.7, 1.0]).to("cuda:0")
            if i == 1700:
                print("Changing target position to [0.7, -0.7, 0.0]")
                rl_task_env.target_position[:] = torch.tensor([-0.7, -0.7, 1.0]).to("cuda:0")
            if i == 2200:
                print("Changing target position to [0.7, 0.7, 0.0]")
                rl_task_env.target_position[:] = torch.tensor([0.7, -0.7, 1.0]).to("cuda:0")
            if i == 2700:
                print("Changing target position to [0.0, 0.0, 0.0]")
                rl_task_env.target_position[:] = torch.tensor([0.0, 0.0, 1.0]).to("cuda:0")
            if i == 200:
                start = time.time()

            # actions = torch.tanh(actions)
            obs, reward, terminated, truncated, info = rl_task_env.step(actions=actions)
            start_time = time.time()
            actions[:] = model.forward(obs["observations"])
            action_list.append(actions[0].cpu().numpy())
            obs_list.append(obs["observations"][0].cpu().numpy())
            euler_angle_list.append(obs_dict["robot_euler_angles"][0].cpu().numpy())
            robot_position_list.append(obs_dict["robot_position"][0].cpu().numpy())
            end_time = time.time()

        state_array = np.array(obs_list[200:])
        action_array = np.array(action_list[200:])
        euler_array = np.array(euler_angle_list[200:])
        robot_position_array = np.array(robot_position_list[200:])

        # Before plotting, create a time array in seconds
        time_in_seconds = np.arange(len(state_array)) / 100.0  # 100 timesteps = 1 second

        # Configure plot style
        plt.rcParams['text.usetex'] = False
        plt.rcParams['figure.figsize'] = (10, 10)  # Width, height in inches
        plt.rcParams['font.size'] = 12  # Base font size
        plt.rcParams['axes.titlesize'] = 14  # Size for axis titles
        plt.rcParams['axes.labelsize'] = 16  # Size for x and y labels
        plt.rcParams['legend.fontsize'] = 16  # Size for legends

        # Create a 3x2 grid of subplots with better spacing
        #fig, axs = plt.subplots(3, 2, constrained_layout=True)

        # Position plot
        # axs[0,0].plot(time_in_seconds, state_array[:, 0:3])
        # axs[0,0].legend([r"$p_x$", r"$p_y$", r"$p_z$"])
        # axs[0,0].set_ylabel(r"Position Error [$m$]")
        # axs[0,0].set_ylim(-1.6, 1.5)
        # axs[0,0].set_yticks(np.arange(-1.5, 1.6, 0.5))
        # # Velocity plot
        # axs[1,0].plot(time_in_seconds, state_array[:, 9:12])
        # axs[1,0].legend([r"$v_x$", r"$v_y$", r"$v_z$"])
        # axs[1,0].set_ylabel(r"Velocity [$\frac{m}{s}$]")
        # axs[1,0].set_ylim(-1.8, 1.5)
        # axs[1,0].set_yticks(np.arange(-1.5, 1.6, 0.5))

        # # Euler angles plot - FIX HERE
        # axs[0,1].plot(time_in_seconds, euler_array)  # Plot on correct subplot
        # axs[0,1].legend([r"$\phi$", r"$\theta$", r"$\psi$"])
        # axs[0,1].set_ylabel(r"Euler Angles [$rad$]")
        # axs[0,1].set_ylim(-0.5, 0.5)
        # axs[0,1].set_yticks(np.arange(-0.4, 0.4, 0.2))

        # axs[1,1].plot(time_in_seconds, state_array[:, 12:15])
        # axs[1,1].set_ylabel(r"Angular Velocity [$\frac{rad}{s}$]")
        # axs[1,1].legend([r"$\omega_x$", r"$ω_y$", r"$ω_z$"])
        # axs[1,1].set_ylim(-3.5, 3.5)
        # axs[1,1].set_yticks(np.arange(-3, 3, 1))
        # # Actions plot - motor thrust
        # action_array = np.clip(action_array, -1, 1)
        # axs[2].plot(time_in_seconds, (action_array)*3 + 3)
        # axs[2].set_xlabel(r"Time [$s$]")  # Changed from steps to s
        # axs[2].set_ylabel(r"Motor Thrust [$N$]")
        # axs[2].legend([r"$u_1$", r"$u_2$", r"$u_3$", r"$u_4$"])
        # axs[2].set_ylim(0, 6)


        fig, axs = plt.subplots(3, constrained_layout=True)

        # Position plot
        axs[0].plot(time_in_seconds, state_array[:, 0:3])
        axs[0].legend([r"$p_x$", r"$p_y$", r"$p_z$"])
        axs[0].set_ylabel(r"Position Error [$m$]")
        axs[0].set_ylim(-1.6, 1.5)
        # Velocity plot
        axs[1].plot(time_in_seconds, state_array[:, 9:12])
        axs[1].legend([r"$v_x$", r"$v_y$", r"$v_z$"])
        axs[1].set_ylabel(r"Velocity [$\frac{m}{s}$]")
        axs[1].set_ylim(-1.8, 1.5)
        # Actions plot - motor thrust
        action_array = np.clip(action_array, -1, 1)
        axs[2].plot(time_in_seconds, (action_array)*3 + 3)
        axs[2].set_xlabel(r"Time [$s$]")  # Changed from steps to s
        axs[2].set_ylabel(r"Motor Thrust [$N$]")
        axs[2].legend([r"$u_1$", r"$u_2$", r"$u_3$", r"$u_4$"])
        axs[2].set_ylim(0, 6)

        # Add a title to the figure
        fig.suptitle("Neural Control in Simulation", fontsize=20)

        # Save with high resolution
        plt.savefig("simulation_plots.png", dpi=300, bbox_inches='tight')
        plt.show()

        data_to_save = np.column_stack((time_in_seconds, robot_position_array))
        np.savetxt("trajectory_data.csv", data_to_save, delimiter=",", header="time,p_x,p_y,p_z", comments='')

    # # TO change setpoints:
    #rl_task_env.target_position[:] = torch.tensor([0.0, 0.0, 1.0]).to("cuda:0")

    end = time.time()