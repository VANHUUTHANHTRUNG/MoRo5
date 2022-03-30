import numpy as np
import matplotlib.pyplot as plt
from visualize_mobile_robot import sim_mobile_robot
from detect_obstacle import detect_obstacle_e5t2

# Constants and Settings
Ts = 0.01  # Update simulation every 10ms
t_max = np.pi  # total simulation duration in seconds
# Set initial state
init_state = np.array([-2., -.5, 0.])  # px, py, theta
IS_SHOWING_2DVISUALIZATION = True

# Define Field size for plotting (should be in tuple)
field_x = (-2.5, 2.5)
field_y = (-2, 2)


# Compute the positions that is detected by the range sensor
# to be used in visualization and in control algorithm
def compute_sensor_endpoint(robot_state, sensors_dist):
    # NOTE: we assume sensor position is in the robot's center
    sens_N = len(sensors_dist)
    obst_points = np.zeros((3, sens_N))

    for i in range(sens_N):
        obst_points[:, i] = robot_state + 0.1 * i * np.ones(3)  # NOTE: this is dummy value
        # TODO: do proper calculation

    return obst_points[:2, :]  # only return x and y values


# MAIN SIMULATION COMPUTATION
# ---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max / Ts)  # Total Step for simulation

    # Initialize robot's state (Single Integrator)
    robot_state = init_state.copy()  # numpy array for [px, py, theta]
    desired_state = np.array([2., 0., 0.])  # numpy array for goal / the desired [px, py, theta]
    current_input = np.array([0., 0., 0.])  # initial numpy array for [vx, vy, omega]

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros((sim_iter, len(robot_state)))
    goal_history = np.zeros((sim_iter, len(desired_state)))
    input_history = np.zeros((sim_iter, len(current_input)))

    if IS_SHOWING_2DVISUALIZATION:  # Initialize Plot
        sim_visualizer = sim_mobile_robot('omnidirectional')  # Omnidirectional Icon
        # sim_visualizer = sim_mobile_robot( 'unicycle' ) # Unicycle Icon
        sim_visualizer.set_field(field_x, field_y)  # set plot area
        sim_visualizer.show_goal(desired_state)
        # ADD OBJECT TO PLOT
        obst_vertices = np.array([[-1., -1.5], [1., -1.5], [1., 1.5], [-1., 1.5], [-1., 1.], [0.5, 1.], [0.5, -1.], [-1., -1.], [-1., -1.5]])
        sim_visualizer.ax.plot(obst_vertices[:, 0], obst_vertices[:, 1], '--r')

        # get sensor reading
        sensors_dist = detect_obstacle_e5t2(robot_state[0], robot_state[1], robot_state[2])
        # compute and plot sensor reading endpoint
        obst_points = compute_sensor_endpoint(robot_state, sensors_dist)
        pl_sens, = sim_visualizer.ax.plot(obst_points[0], obst_points[1], '.')
        pl_txt = [sim_visualizer.ax.text(obst_points[0, i], obst_points[1, i], str(i)) for i in
                  range(len(sensors_dist))]

    for it in range(sim_iter):
        # record current state at time-step t
        state_history[it] = robot_state
        goal_history[it] = desired_state

        # Get information from sensors
        sensors_dist = detect_obstacle_e5t2(robot_state[0], robot_state[1], robot_state[2])
        # compute and plot sensor reading endpoint
        obst_points = compute_sensor_endpoint(robot_state, sensors_dist)

        # IMPLEMENTATION OF CONTROLLER
        # ------------------------------------------------------------
        # Compute the control input 
        current_input[0] = 0
        current_input[1] = 0
        current_input[2] = -2.

        # TODO: change the implementation to switching
        # ------------------------------------------------------------

        # record the computed input at time-step t
        input_history[it] = current_input

        if IS_SHOWING_2DVISUALIZATION:  # Update Plot
            sim_visualizer.update_time_stamp(it * Ts)
            sim_visualizer.update_goal(desired_state)
            sim_visualizer.update_trajectory(state_history[:it + 1])  # up to the latest data
            # update sensor visualization
            pl_sens.set_data(obst_points[0], obst_points[1])
            for i in range(len(sensors_dist)): pl_txt[i].set_position((obst_points[0, i], obst_points[1, i]))

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
    return state_history, goal_history, input_history


if __name__ == '__main__':
    # Call main computation for robot simulation
    state_history, goal_history, input_history = simulate_control()

    # ADDITIONAL PLOTTING
    # ----------------------------------------------
    t = [i * Ts for i in range(round(t_max / Ts))]

    # # Plot historical data of control input
    # fig2 = plt.figure(2)
    # ax = plt.gca()
    # ax.plot(t, input_history[:,0], label='vx [m/s]')
    # ax.plot(t, input_history[:,1], label='vy [m/s]')
    # ax.plot(t, input_history[:,2], label='omega [rad/s]')
    # ax.set(xlabel="t [s]", ylabel="control input")
    # plt.legend()
    # plt.grid()

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

    plt.show()
