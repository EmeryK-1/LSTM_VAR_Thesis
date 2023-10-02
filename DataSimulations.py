import numpy as np
import pandas as pd


def generate_trend(c, a, time_steps):
    """
    Generate a trend with a given intercept and slope.
    :param c: intercept
    :param a: slope
    :param time_steps: number of time steps
    :return: trend component
    """
    t = np.arange(0, time_steps)[:, np.newaxis]
    return pd.DataFrame(np.array(a * t + c), columns=['Variable_{}'.format(i) for i in range(len(a))])


def generate_seasonality(f, s, time_steps):
    """
    Generate a seasonality with a given frequency and amplitude.
    :param f: frequency
    :param s: scale
    :param time_steps: number of time steps
    :return: seasonality component
    """
    t = np.arange(0, time_steps)[:, np.newaxis]
    return pd.DataFrame(np.sin((2 * np.pi / f) * (t % f)) * s, columns=['Variable_{}'.format(i) for i in range(len(f))])


def generate_interrelations(time_steps, n_variables, companion_matrix, noise=False):
    """
    Simulate a Vector Autoregressive (VAR) model.

    Parameters:
    time_steps (int): The number of time steps.
    n_variables (int): The number of variables.
    companion_matrix (np.ndarray): The VAR companion matrix.
    noise (bool or np.ndarray): Whether to add random noise, or a specific noise array.

    Returns:
    pd.DataFrame: A DataFrame with the generated interrelations.
    """
    # If companion matrix is none, return 0s
    if companion_matrix is None:
        interrelations = pd.DataFrame(np.zeros((time_steps, n_variables)), columns=['Variable_{}'.format(i) for i in range(n_variables)])
        return interrelations, np.zeros((time_steps, n_variables)) if isinstance(noise, bool) and noise else None
    # Determine the number of lags from the companion matrix
    p = companion_matrix.shape[0] // n_variables

    # Create the initial state with zeros
    initial_state = np.zeros(p * n_variables)

    # Prepare for simulation
    interrelations = [initial_state[n_variables * i: n_variables * (i + 1)] for i in range(p)]
    if isinstance(noise, bool) and noise:
        noise = np.random.normal(size=(time_steps, n_variables))

    # Simulate the time series
    for i in range(time_steps):
        initial_state = np.dot(companion_matrix, initial_state)
        initial_state[:n_variables] += np.random.normal(size=n_variables) if isinstance(noise, bool) else noise[i]
        interrelations.append(initial_state[:n_variables])

    interrelations = pd.DataFrame(np.array(interrelations), columns=['Variable_{}'.format(i) for i in range(n_variables)])
    # Drop first p rows, as they are just the initial state
    interrelations = interrelations.iloc[p:].reset_index(drop=True)
    return interrelations, None if isinstance(noise, bool) and not noise else noise


def to_companion(coefficients):
    """
    Convert a set of coefficients to its companion matrix form.
    :param coefficients: coefficients
    :return: companion matrix
    """
    m = coefficients.shape[0]
    p = coefficients.shape[1] // m
    # Stack coeffs and i, fill rest with 0
    coefficients = np.vstack((coefficients, np.zeros((m*(p-1), m*p))))
    # Set bottom left (p-1)x(p-1) square to be identity
    coefficients[m:, :m*(p-1)] = np.eye(m*(p-1))
    return coefficients


def make_stable(companion_matrix, num_time_series, num_lags):
    """
    Make a companion matrix stable by reducing the eigenvalues to 0.95.
    :param companion_matrix: companion_matrix
    :param num_time_series: number of time series
    :param num_lags: number of lags
    :return: stable companion matrix
    """
    while np.max(np.abs(np.linalg.eigvals(companion_matrix))) > 0.95:
        companion_matrix = companion_matrix * 0.95
        companion_matrix[num_time_series:, :num_time_series*(num_lags-1)] = np.eye(num_time_series*(num_lags-1))
    return companion_matrix


def simulate(time_steps, m, p, seed=None):
    """
    Simulate a multivariate dataset with a given number of time steps, variables and lags.
    Computes: trend, seasonality, var, var+trend, var+seasonality, trend+seasonality, var+trend+seasonality
    :param time_steps: number of time steps
    :param m: number of variables
    :param p: number of lags in the VAR data
    :param seed: seed for random number generator
    :return: dictionary with the simulated data
    """
    if seed is not None:
        np.random.seed(seed)

    c = np.random.normal(0, 1, m)
    a = np.random.normal(0, 1, m)

    s = np.random.uniform(2, 50, m)
    f = np.random.uniform(5, 30, m)
    companion_matrix = make_stable(to_companion(np.random.normal(0, 1, (m, m * p))), m, p)

    trend = generate_trend(c, a, time_steps)
    seasonality = generate_seasonality(f, s, time_steps)
    var = generate_interrelations(time_steps, m, companion_matrix)[0]

    return {'t': trend, 's':seasonality, 'v':var, 'vt':var+trend, 'vs':var+seasonality,'ts':trend+seasonality,'vts':var+trend+seasonality, 'parameters': {'n':time_steps, 'm':m, 'p':p, 'seed':seed, 'c': c, 'a': a, 's': s, 'f': f, 'companion_matrix': companion_matrix}}
