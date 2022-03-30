import numpy as np
import matplotlib.pyplot as plt
from visualize_mobile_robot import sim_mobile_robot

# Constants and Settings
Ts = 0.01 # Update simulation every 10ms
t_max = np.pi*3 # total simulation duration in seconds
K = 3 # gain
# V0 = 3
# BETA = 5
# Set initial state
init_state = np.array([0., 0., 0.]) # px, py, theta
IS_SHOWING_2DVISUALIZATION = True

# Define Field size for plotting (should be in tuple)
field_x = (-3, 3)
field_y = (-3, 3)


# MAIN SIMULATION COMPUTATION
#---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max/Ts) # Total Step for simulation

    # Initialize robot's state (Single Integrator)
    robot_state = init_state.copy() # numpy array for [px, py, theta]
    desired_state = np.array([0., 0., 0.]) # numpy array for goal / the desired [px, py, theta]
    # desired_state = np.array([np.cos(2*Ts*it), np.sin(2*Ts*it), 0])
    current_input = np.array([0., 0., 0.]) # initial numpy array for [vx, vy, omega]

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros( (sim_iter, len(robot_state )) ) 
    goal_history = np.zeros( (sim_iter, len(desired_state )) ) 
    input_history = np.zeros( (sim_iter, len(current_input )) ) 
    error_history = np.zeros( (sim_iter, len(desired_state )))

    if IS_SHOWING_2DVISUALIZATION: # Initialize Plot
        sim_visualizer = sim_mobile_robot( 'omnidirectional' ) # Omnidirectional Icon
        #sim_visualizer = sim_mobile_robot( 'unicycle' ) # Unicycle Icon
        sim_visualizer.set_field( field_x, field_y ) # set plot area
        sim_visualizer.show_goal(desired_state)

    for it in range(sim_iter):
        # record current state at time-step t
        desired_state = np.array([np.cos(2*Ts*it), np.sin(2*Ts*it), 0])
        state_history[it] = robot_state
        goal_history[it] = desired_state

        # IMPLEMENTATION OF CONTROLLER
        #------------------------------------------------------------
        # Compute the control input

        # proportional control - static gain
        # current_input[:2] = K*(desired_state[:2]-robot_state[:2])
        # theta_error = desired_state[2] - robot_state[2]
        # current_input[2] = K*((theta_error + np.pi)%(2*np.pi) - np.pi)

        # proportional control with time-varying gain
        #------------------------------------------------------------
        error = desired_state - robot_state
        error[2] = (error[2] + np.pi)%(2*np.pi) - np.pi
        forward_part = np.array([-2*np.sin(2*Ts*it), 2*np.cos(2*Ts*it), 0])
        # error_norm = np.linalg.norm(error)
        # K = V0*(1 - np.exp(-BETA*error_norm))/(error_norm + 1E-10) # add small eps to avoid division by 0 if it happens
        current_input = K*error + forward_part

        # record the computed input at time-step t
        input_history[it] = current_input
        error_history[it] = error
        if IS_SHOWING_2DVISUALIZATION: # Update Plot
            sim_visualizer.update_time_stamp( it*Ts )
            sim_visualizer.update_goal( desired_state )
            sim_visualizer.update_trajectory( state_history[:it+1] ) # up to the latest data
        
        #--------------------------------------------------------------------------------
        # Update new state of the robot at time-step t+1
        # using discrete-time model of single integrator dynamics for omnidirectional robot
        robot_state = robot_state + Ts*current_input # will be used in the next iteration
        robot_state[2] = ( (robot_state[2] + np.pi) % (2*np.pi) ) - np.pi # ensure theta within [-pi pi]

        # Update desired state if we consider moving goal position
        #desired_state = desired_state + Ts*(-1)*np.ones(len(robot_state))

    # End of iterations
    # ---------------------------
    # return the stored value for additional plotting or comparison of parameters
    return state_history, goal_history, input_history, error_history


if __name__ == '__main__':
    
    # Call main computation for robot simulation
    state_history, goal_history, input_history, error_history = simulate_control()


    # ADDITIONAL PLOTTING
    #----------------------------------------------
    t = [i*Ts for i in range( round(t_max/Ts) )]

    # # Plot historical data of control input
    fig2 = plt.figure(2)
    ax = plt.gca()
    ax.plot(t, input_history[:,0], label='vx [m/s]', alpha=0.5)
    ax.plot(t, input_history[:,1], label='vy [m/s]', alpha=0.5)
    ax.plot(t, input_history[:,2], label='omega [rad/s]', alpha=0.5)
    ax.set(xlabel="t [s]", ylabel="control input")
    plt.legend()
    plt.grid()

    # Plot historical data of state
    fig3 = plt.figure(3)
    ax = plt.gca()
    ax.plot(t, state_history[:,0], label='px [m]', alpha=0.5)
    ax.plot(t, state_history[:,1], label='py [m]', alpha=0.5)
    ax.plot(t, state_history[:,2], label='theta [rad]', alpha=0.5)
    ax.plot(t, goal_history[:,0], ':', label='goal px [m]', alpha=0.5)
    ax.plot(t, goal_history[:,1], ':', label='goal py [m]', alpha=0.5)
    ax.plot(t, goal_history[:,2], ':', label='goal theta [rad]', alpha=0.5)
    ax.set(xlabel="t [s]", ylabel="state")
    plt.legend()
    plt.grid()

    fig4 = plt.figure(4)
    ax = plt.gca()
    ax.plot(t, error_history[:, 0], label='error x-coord', alpha=0.5)
    ax.plot(t, error_history[:, 1], label='error y-coord', alpha=0.5)
    ax.set(xlabel="t [s]", ylabel="error")
    plt.legend()
    plt.grid()

    plt.show()
