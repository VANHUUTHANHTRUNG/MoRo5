import numpy as np
import matplotlib.pyplot as plt
from visualize_mobile_robot import sim_mobile_robot
import math

# Constants and Settings
Ts = 0.01  # Update simulation every 10ms
t_max = np.pi * 4  # total simulation duration in seconds

# Set initial stat
init_state = np.array([-2, -.5, 0.])  # px, py, theta
obstacle_state = np.array([0, 0, 0.5])  # px_o, y_o, radius
obstacle_radius = 0.5

# Robot's constraint
v_trans_max = 0.5  # m/s
v_rot_max = 5  # rad/s
robot_radius = 0.21  # robot's radius in meter
d_safe = obstacle_radius + robot_radius + 0.1  # safe distance as sum of 2 radiuses and a small margin of 0.01

# Parameters to adjust
epsilon = 1E-2
c = v_trans_max / 100 / epsilon
beta = 5

IS_SHOWING_2DVISUALIZATION = True

# Define Field size for plotting (should be in tuple)
field_x = (-2.5, 2.5)
field_y = (-2, 2)


# MAIN SIMULATION COMPUTATION
# ---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max / Ts)  # Total Step for simulation

    # Initialize robot's state (Single Integrator)
    robot_state = init_state.copy()  # numpy array for [px, py, theta]
    desired_state = np.array([2, 1., 0])  # numpy array for goal / the desired [px, py, theta] static
    current_input = np.array([0., 0., 0.])  # initial numpy array for [vx, vy, omega]
    error_history = np.zeros((sim_iter, len(desired_state)))

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros((sim_iter, len(robot_state)))
    goal_history = np.zeros((sim_iter, len(desired_state)))
    input_history = np.zeros((sim_iter, len(current_input)))
    obstacle_dist_history = np.zeros((sim_iter, 1))

    if IS_SHOWING_2DVISUALIZATION:  # Initialize Plot
        sim_visualizer = sim_mobile_robot('omnidirectional')  # Omnidirectional Icon
        # sim_visualizer = sim_mobile_robot( 'unicycle' ) # Unicycle Icon
        sim_visualizer.set_field(field_x, field_y)  # set plot area
        sim_visualizer.show_goal(desired_state)
        # Plot obstacle
        outer_circle = plt.Circle((obstacle_state[0], obstacle_state[1]), obstacle_radius+0.1, color='gold')
        inner_circle = plt.Circle((obstacle_state[0], obstacle_state[1]), obstacle_radius, color='red')
        sim_visualizer.ax.add_artist(outer_circle)
        sim_visualizer.ax.add_artist(inner_circle)

    for it in range(sim_iter):
        # record current state at time-step t
        state_history[it] = robot_state
        goal_history[it] = desired_state

        # IMPLEMENTATION OF CONTROLLER
        # ------------------------------------------------------------
        # Compute the control input
        distance = np.linalg.norm(robot_state[:2] - obstacle_state[:2])
        # print(f'Distance: {distance}, Limit: {d_safe+epsilon}')
        if distance > d_safe + epsilon:
            error = desired_state - robot_state
            error[2] = (error[2] + np.pi) % (2 * np.pi) - np.pi
            error_norm = np.linalg.norm(error[:2])
            K_g = v_trans_max * (1 - np.exp(-beta * error_norm)) / (error_norm + 1E-10)
            if error_norm >= 0.005:
                current_input = K_g * error
            else:
                break   # Close enough to the desired position -> Stop
        else:
            error = robot_state - obstacle_state
            error[2] = math.atan2(error[1], error[0])
            error[2] = (error[2] + np.pi) % (2 * np.pi)
            K_o = (1 / distance) * (c / (distance ** 2 + epsilon))
            current_input = K_o * error

        # Adjust input based on robot's physical limitation
        v_ratio = current_input[0] / (current_input[2] + current_input[1])
        current_input[1] = 0.5 / np.sqrt(1 + v_ratio ** 2)  # Scale vx, vy based on max velocity
        current_input[0] = v_ratio * current_input[1]
        current_input[2] = np.min([current_input[2], 5 * Ts])  # Upper limit of possible angular difference
        # ------------------------------------------------------------

        # record the computed input at time-step t
        input_history[it] = current_input
        error_history[it] = error
        obstacle_dist_history[it] = distance

        if IS_SHOWING_2DVISUALIZATION:  # Update Plot
            sim_visualizer.update_time_stamp(it * Ts)
            sim_visualizer.update_goal(desired_state)
            sim_visualizer.update_trajectory(state_history[:it + 1])  # up to the latest data

        # --------------------------------------------------------------------------------
        # Update new state of the robot at time-step t+1
        # using discrete-time model of single integrator dynamics for omnidirectional robot
        robot_state = robot_state + Ts * current_input  # will be used in the next iteration
        robot_state[2] = ((robot_state[2] + np.pi) % (2 * np.pi)) - np.pi  # ensure theta within [-pi pi]

        # Update desired state if we consider moving goal position
        # desired_state = desired_state + Ts*(-1)*np.ones(len(robot_state))

    # End of iterations
    # ---------------------------
    # return the stored value for additional plotting or comparison of parameters
    return state_history, goal_history, input_history, obstacle_dist_history, error_history


if __name__ == '__main__':
    # Call main computation for robot simulation
    state_history, goal_history, input_history, obstacle_dist_history, error_history = simulate_control()

    # ADDITIONAL PLOTTING
    # ----------------------------------------------
    t = [i * Ts for i in range(round(t_max / Ts))]

    # # Plot historical data of control input
    fig2 = plt.figure(2)
    ax = plt.gca()
    ax.plot(t, np.sqrt(np.power(input_history[:,0],2) + np.power(input_history[:,1],2)), label='v [m/s]')
    ax.plot(t, input_history[:,2], label='omega [rad/s]')
    ax.set(xlabel="t [s]", ylabel="control input")
    plt.legend()
    plt.grid()

    # Plot historical data of state
    fig3 = plt.figure(3)
    ax = plt.gca()
    ax.plot(t, state_history[:, 0], label='px [m]')
    ax.plot(t, state_history[:, 1], label='py [m]')
    ax.plot(t, state_history[:, 2], label='theta [rad]')
    ax.plot(t, goal_history[:, 0], ':', label='goal px [m]')
    ax.plot(t, goal_history[:, 1], ':', label='goal py [m]')
    ax.plot(t, goal_history[:, 2], ':', label='goal theta [rad]')
    ax.set(xlabel="t [s]", ylabel="state")
    plt.legend()
    plt.grid()

    fig4 = plt.figure(4)
    ax = plt.gca()
    ax.plot(t, obstacle_dist_history, label='||x - x_o|| [m]')
    ax.set(xlabel="t [s]", ylabel="distance to obstacle")
    plt.legend()
    plt.grid()

    fig5 = plt.figure(5)
    ax = plt.gca()
    ax.plot(t, error_history[:,0], label='Error x [m]')
    ax.plot(t, error_history[:,1], label='Error y [m]')
    ax.set(xlabel="t [s]", ylabel="Error")
    plt.legend()
    plt.grid()

    plt.show()
