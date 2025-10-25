import numpy as np
from scipy.integrate import solve_ivp
import config
from controller import design_lqr_controller
from dynamics import inverted_pendulum_dynamics

def generate_training_data():
    """
    Simulates the inverted pendulum with an LQR controller and saves the data
    """
    print("--- Generating Training Data ---")
    # Design the controller
    K = design_lqr_controller(config)

    # Stabilize around the upright position
    equilibrium_state = np.array([0, 0, np.pi, 0])

    # Define the control law u = -Kx
    def control_law(x_state):
        error_state = x_state - equilibrium_state
        return (-K @ error_state).item()

    # The dynamics function for solve_ivp needs to use the controller
    def controlled_system(t, x_state):
        return inverted_pendulum_dynamics(t, x_state, u_func=control_law, params=config)
    
    # Time vector for simulation
    t_span = [0, config.T_END]
    t_eval = np.arange(0, config.T_END, config.DT)

    # Run the simulation
    print(f"Simulating from initial state: {config.X0}")
    sol = solve_ivp(
        controlled_system, 
        t_span, 
        config.X0, 
        t_eval=t_eval, 
        dense_output=True
    )
    
    # Extract states and calculate control inputs used
    x_train = sol.y.T  # Transpose to get (n_samples, n_features)
    t_train = sol.t
    
    # Recompute the control inputs used at each time step
    u_train = np.array([control_law(state) for state in x_train]).reshape(-1, 1)

    # Save the data
    np.savez(
        config.TRAIN_DATA_PATH,
        x=x_train,
        u=u_train,
        t=t_train
    )
    print(f"Training data successfully generated and saved to {config.TRAIN_DATA_PATH}")

    return x_train, u_train, t_train
