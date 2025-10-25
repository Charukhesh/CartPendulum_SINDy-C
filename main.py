import numpy as np
import generate_data
import discover_model
import validate_model
import visualise
import config
import matplotlib.pyplot as plt

def main():
    """
    Main function to run the complete SINDy pipeline for the inverted pendulum.
    """
    # Generate training data from a controlled simulation
    generate_data.generate_training_data()

    data = np.load(config.TRAIN_DATA_PATH)
    t = data['t']
    x_states = data['x']

    # print("\nDisplaying animation of the controlled (training) phase...")
    #visualise.animate_pendulum(t, x_states, save_path=config.TRAIN_GIF_PATH)

    print("Displaying state variable plots...")
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    state_labels = [
        r'Cart Position $x$ (m)', 
        r'Cart Velocity $\dot{x}$ (m/s)', 
        r'Pendulum Angle $\theta$ (rad)', 
        r'Pendulum Angular Velocity $\dot{\theta}$ (rad/s)'
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Blue, Orange, Green, Red
    axs_flat = axs.flatten()
    
    for i in range(4):
        ax = axs_flat[i] 
        ax.plot(t, x_states[:, i], color=colors[i], linewidth=2)
        ax.set_title(state_labels[i], fontsize=12)
        ax.grid(True)
        
        if i == 2:
            ax.axhline(np.pi, color='black', linestyle='--', label=r'$\pi$ (Setpoint)')
            ax.legend()
            ax.set_ylabel("Angle (rad)")
        else:
            ax.axhline(0, color='black', linestyle='--', label=r'$0$ (Setpoint)')
            ax.legend()
        if i >= 2:
            ax.set_xlabel("Time (s)")

    fig.suptitle("State Trajectories During Controlled (Training) Phase", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Phase plot of the pendulum's angle vs. angular velocity
    plt.figure(figsize=(8, 6))
    plt.plot(x_states[:, 2], x_states[:, 3])
    # Red dot at the starting point and a green dot at the equilibrium point
    plt.plot(x_states[0, 2], x_states[0, 3], 'ro', markersize=10, label='Start')
    plt.plot(np.pi, 0, 'go', markersize=10, label='Equilibrium (Setpoint)')
    plt.title("Pendulum Phase Plot During Controlled (Training) Phase")
    plt.xlabel("Pendulum Angle $\\theta$ (rad)")
    plt.ylabel(r"Pendulum Angular Velocity $\dot{\theta}$ (rad/s)")
    plt.grid(True)
    plt.legend()
    plt.axis('equal') # Helps to visualize the dynamics more intuitively
    plt.show()
    
    # Use SINDy to discover the governing equations from the data
    # sindy_model = discover_model.discover_governing_equations()
    
    # Validate the discovered model on an uncontrolled, unseen scenario
    # validate_model.validate_discovered_model(sindy_model)

if __name__ == "__main__":
    main()