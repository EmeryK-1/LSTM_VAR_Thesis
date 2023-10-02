import os

import numpy as np
import pandas as pd

import Evaluation
from Models import VAR


def save(df, file, folder):
    """
    Save the dataframe to a csv file in the folder
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    full_path = os.path.join(folder, file + '.csv')
    df.fillna(0).to_csv(full_path, index=False)


def get_var_eval(dataset, parameters):
    """
    Train and get the evaluation metrics for the VAR model
    :param dataset: DataSet object
    :param parameters: parameters of data generation
    :return: Dictionary of evaluation metrics
    """
    var_coefs = VAR.get_var_coefficients(dataset.train, parameters['p'])
    var_eval = Evaluation.matrix_rmse(var_coefs, parameters['companion_matrix'][:parameters['m']])
    var_eval['test_rmse'] = np.mean((VAR.predict(var_coefs, dataset.test) - dataset.test[parameters['p']:]) ** 2)
    return var_eval


def rename_columns(df):
    """
    Rename the columns of the dataframe to be more readable
    :param df: Dataframe with columns to be renamed
    :return: df with renamed columns
    """
    df = df.rename(columns={'loss': 'Loss', 'test_rmse': 'Test RMSE', 'val_loss': 'Validation Loss', 'mse': 'Test RMSE',
                            'all': 'Matrix RMSE', 'model': 'Model', 'simulation': 'Simulation', 'epoch': 'Epoch',
                            'sign': 'Same Sign %', 'variable': 'Variable'})
    return df


def order_df(df, order):
    """
    Re-order the dataframe to be in the order of the models
    :param df: Dataframe with a Model column
    :param order: List of models in the order they should be
    :return: df with the models in the correct order
    """
    order_mapping = {unit: idx for idx, unit in enumerate(order)}
    df['sorting_key'] = df['Model'].map(order_mapping)
    if 'Simulation' in df.columns:
        df['sorting_key'] = df['sorting_key'] + df['Simulation'] * 0.1
    df = df.sort_values(by='sorting_key')
    df = df.drop(columns='sorting_key')
    return df


def rename_variables(df):
    """
    Rename the gradient variables to be more readable
    :param df: Gradients df
    :return: df with renamed variables
    """
    v = df['Variable'].unique()
    for i in v:
        name = i.split(':')[0].split('/')
        if len(name) != 1:
            t1 = name[1]
            t2 = name[-1]
            t2 = t2.split('_')
            t2 = [t.capitalize() for t in t2]
            t2 = ' '.join(t2)
            name = t1+' '+t2
        else:
            name = name[0]
        df.loc[df['Variable'] == i, 'Variable'] = name
    return df


def compute_convergence(df, patience=10, threshold=0.1):
    """
    Compute the convergence time for each model using early stopping of *threshold* with *patience* for both the
    validation loss and matrix rmse
    :param df: Dataframe with all results
    :param patience: Number of epochs needed to be within threshold to be considered converged
    :param threshold: Threshold for convergence
    :return: df with convergence times for each model
    """
    def stopping_criteria(losses, min_epochs=50, check_interval=5, patience=10, threshold=0.1):
        if len(losses) < min_epochs:
            return None

        # Calculate smoothed losses using simple moving average
        smoothed_losses = [sum(losses[max(0, i - check_interval):i + 1]) / len(losses[max(0, i - check_interval):i + 1])
                           for i in range(len(losses))]
        for i in range(len(smoothed_losses) - patience):
            # Check if loss has stabilized for 'patience' epochs
            differences = [((smoothed_losses[j] - smoothed_losses[j + 1]) / smoothed_losses[j] * 100) for j in
                           range(i, i + patience - 1)]
            if max(differences) < threshold or np.mean(differences) < 0:
                return i + patience - 1  # Return True and the epoch number where it is believed to have converged

        return len(losses)

    converge = []
    for combo in df.groupby(['Model', 'Simulation']).first().index.tolist():
        converge.append([combo[0], combo[1],
                         stopping_criteria(df[(df['Model'] == combo[0]) & (df['Simulation'] == combo[1])][
                                               'Validation Loss'].tolist(), patience=patience, threshold=threshold),
                         stopping_criteria(
                             df[(df['Model'] == combo[0]) & (df['Simulation'] == combo[1])]['Matrix RMSE'].tolist(),
                             patience=patience, threshold=threshold)])
    converge = pd.DataFrame(converge, columns=['Model', 'Simulation', 'Validation Convergence', 'Matrix Convergence'])
    converge['Ratio'] = (converge['Matrix Convergence']) / converge['Validation Convergence']
    return converge


def get_converged(df, converge_time, models):
    """
    Get the converged results for each model
    :param df: Dataframe with all results
    :param converge_time: df with convergence times
    :param models: list of models
    :return: df with converged results
    """
    c = []
    for idx, row in converge_time.reset_index().iterrows():
        c.append(df[(df['Model'] == row['Model']) & (df['Simulation'] == row['Simulation']) & (
                    df['Epoch'] == int(row.values[2]))])
    return order_df(pd.concat(c), models).reset_index(drop=True)
