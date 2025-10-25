# validate_model.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import config
from dynamics import inverted_pendulum_dynamics

def validate_discovered_model(sindy_model):
    """
    Validates the SINDy model by comparing its prediction on an unseen trajectory
    against the true system dynamics.
    """
    print("\n--- Step 3: Validating Discovered Model ---")

    # 1. Define an unseen, challenging initial condition (uncontrolled)
    x0_test = [0, 0, np.pi / 2, 0]  # Start from 90 degrees
    t_end_test = 10
    t_test = np.arange(0, t_end_test, config.DT)

    print(f"Validation scenario: No control, initial state = {x0_test}")
    
    # 2. Simulate the TRUE system dynamics for this test case
    def true_system_uncontrolled(t, x_state):
        return inverted_pendulum_dynamics(t, x_state, u_func=None, params=config)
    
    sol_true = solve_ivp(
        true_system_uncontrolled, 
        [0, t_end_test], 
        x0_test, 
        t_eval=t_test
    )
    x_true = sol_true.y.T

    # 3. Simulate the DISCOVERED SINDy model for the same test case
    x_sindy = sindy_model.simulate(x0_test, t_test, u=lambda t: 0)

    # 4. Plot the comparison
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    state_labels = ["x", "x_dot", "theta", "theta_dot"]
    
    for i in range(4):
        axs[i].plot(t_test, x_true[:, i], 'k-', label='True System', linewidth=2)
        axs[i].plot(t_test, x_sindy[:, i], 'r--', label='SINDy Model', linewidth=2)
        axs[i].set_ylabel(state_labels[i])
        axs[i].grid(True)
        axs[i].legend()

    axs[3].set_xlabel("Time (s)")
    fig.suptitle("Validation: True Dynamics vs. Discovered SINDy Model (Uncontrolled)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
