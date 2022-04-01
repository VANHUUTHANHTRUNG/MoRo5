import numpy as np
import matplotlib.pyplot as plt
from visualize_mobile_robot import sim_mobile_robot
from detect_obstacle import detect_obstacle_e5t2

# Constants and Settings
Ts = 0.01  # Update simulation every 10ms
t_max = np.pi*10  # total simulation duration in seconds
# Set initial state
init_state = np.array([-2., -.5, 0.])  # px, py, theta
IS_SHOWING_2DVISUALIZATION = True

# Robot's constraint
v_trans_max = 0.5  # m/s
v_rot_max = 5  # rad/s
robot_radius = 0.21  # robot's radius in meter

# params
d_safe = robot_radius + 0.01
eps = 0.05
dist_at_cross = 69420

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
        # Update detected points from sensors
        sensor_angle = 2 * np.pi * i / sens_N + robot_state[2]
        obst_points[0, i] = sensors_dist[i] * np.cos(sensor_angle) + robot_state[0]
        obst_points[1, i] = sensors_dist[i] * np.sin(sensor_angle) + robot_state[1]
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
        # Get x_o
        shortest_dist = obst_points[:, np.argsort(sensors_dist)[-2:]]
        X_o = (shortest_dist[0] + shortest_dist[1])/2

        U_wf_t = (shortest_dist[0] - shortest_dist[1])/np.linalg.norm(shortest_dist[0] - shortest_dist[1])
        U_wf_p = (shortest_dist[0] - robot_state[:2]) - np.dot((np.dot(shortest_dist[0] - robot_state[:2], U_wf_t)), U_wf_t)
        U_wf_p = U_wf_p - d_safe*(U_wf_p/np.linalg.norm(U_wf_p))
        U_wf = U_wf_p + U_wf_t

        # print(f'Dist: {np.linalg.norm(robot_state[:2] - X_o)} Limit: {np.abs(d_safe + eps)}')
        print(U_wf.shape)

        if (np.linalg.norm(robot_state[:2] - X_o) <= np.abs(d_safe + eps)): #  and (np.dot(U_wf, desired_state[:2]-robot_state[:2])>0)
            print('wall')
            current_input[0] = U_wf[0]
            current_input[1] = U_wf[1]
        else:
            current_input[0] = 0.7
            current_input[1] = 0
            # current_input[2] = -2

        # Compute the control input 
        

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
