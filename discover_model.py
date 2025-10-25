import numpy as np
import pysindy as ps
import config

def discover_governing_equations():
    """
    Loads the training data and uses SINDy to discover the governing equations
    """
    print("\n--- Step 2: Discovering Model with SINDy ---")
    
    # Load data
    data = np.load(config.TRAIN_DATA_PATH)
    x_train = data['x']
    u_train = data['u']
    t_train = data['t']

    # Define the candidate feature library
    # We need polynomials and trigonometric functions
    poly_library = ps.PolynomialLibrary(degree=2)
    fourier_library = ps.FourierLibrary(n_frequencies=2)
    
    # We also need to define custom functions
    def sin_theta(x): return np.sin(x[:, 2])
    def cos_theta(x): return np.cos(x[:, 2])
    def sin_theta_dot_sq(x): return np.sin(x[:, 2]) * x[:, 3]**2

    custom_funcs = [sin_theta, cos_theta, sin_theta_dot_sq]
    custom_names = [
        lambda x: f'sin({x})',
        lambda x: f'cos({x})',
        lambda x: f'sin({x[2]})*{x[3]}^2'
    ]
    custom_library = ps.CustomLibrary(library_functions=custom_funcs, function_names=custom_names)

    # Combine libraries: Polynomials up to degree 2, sin/cos of theta
    combined_library = ps.GeneralizedLibrary([
        poly_library,
        custom_library
    ])

    # Instantiate SINDy
    optimizer = ps.STLSQ(threshold=config.SINDY_THRESHOLD, alpha=config.SINDY_ALPHA)
    model = ps.SINDy(
        optimizer=optimizer,
        feature_library=combined_library,
        feature_names=["x", "x_dot", "theta", "theta_dot"]
    )
    
    # Fit the model to the data
    print("Fitting SINDy model...")
    model.fit(x=x_train, u=u_train, t=t_train)
    
    # Print the discovered model
    print("Discovered model equations:")
    model.print()
    
    return model