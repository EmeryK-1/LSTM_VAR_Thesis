import math
import numpy as np
import pandas as pd
from scipy import stats


def apply_significance(val, p):
    """
    Apply significance to value
    :param val: value
    :param p: significance level
    :return: updated value
    """
    if math.isnan(val):
        return '-'
    val_s = str(val)
    val_s = val_s + (5 - len(str(val))) * '0'
    if val <= p:
        return '*' + val_s
    else:
        return val_s


def converge_to_latex_table(converge, order):
    """
    Pivots convergence table to latex table that contains mean and std of validation and matrix convergence
    :param converge: table of convergence values
    :param order: order to display models
    :return: pivoted table
    """
    table = converge[['Model', 'Validation Convergence', 'Matrix Convergence', 'Ratio']].groupby('Model').describe()
    # Get mean and std columns for Validation Convergence, Matrix Convergence, Difference
    table = table[
        [('Validation Convergence', 'mean'), ('Validation Convergence', 'std'), ('Matrix Convergence', 'mean'),
         ('Matrix Convergence', 'std'), ('Ratio', 'mean'), ('Ratio', 'std')]]
    table = table.reindex(order)
    table = table.round(2)
    print('Means:', table.mean())
    # Loop through level 0
    for col in table.columns.levels[0]:
        table[(col, 'mean')] = table[(col, 'mean')].astype(str) + ' (' + table[(col, 'std')].astype(str) + ')'
        table = table.drop(columns=(col, 'std'))
    # Remove multiindex, keep level 0
    table.columns = table.columns.droplevel(1)
    return table


def results_to_latex_table(results, lstm_vars, vars):
    """
    Pivots results table to latex table that contains mean and std of matrix rmse, test rmse, and same sign %
    :param results: converged results
    :param lstm_vars: first set of models
    :param vars: second set of models
    :return: pivoted table
    """
    results_desc = results.groupby('Model').describe().reindex(vars+lstm_vars)
    for col in results_desc.columns.levels[0]:
        results_desc = results_desc.drop(
            columns=[(col, 'count'), (col, 'min'), (col, '25%'), (col, '50%'), (col, '75%'), (col, 'max')])
    # Keep only Matrix Euclidean Distance and Mean Squared Error mean and std
    table = results_desc[[('Matrix RMSE', 'mean'), ('Matrix RMSE', 'std'), ('Test RMSE', 'mean'), ('Test RMSE', 'std'),
                          ('Same Sign %', 'mean'), ('Same Sign %', 'std')]].copy()

    for col in ['Matrix RMSE', 'Test RMSE', 'Same Sign %']:

        table[(col, 'Mean')] = np.round(table[(col, 'mean')], 3 if col == 'Matrix RMSE' else 3).astype(
            str) + ' (' + np.round(table[(col, 'std')], 3 if col == 'Matrix RMSE' else 3).astype(str) + ')'
        if len(vars) == len(lstm_vars):
            table.loc[lstm_vars, (col, 'Change (vs VAR)')] = np.round(
                (table.loc[vars, (col, 'mean')].values - table.loc[lstm_vars, (col, 'mean')].values) / table.loc[
                    vars, (col, 'mean')].values * 100, 3)
            table.loc[lstm_vars, (col, 'Change (vs VAR)')] = table.loc[lstm_vars, (col, 'Change (vs VAR)')].astype(
                str) + '\%'
            table = table.fillna('-')
        elif len(vars) == 1:
            value = table[(col, 'mean')][vars[0]]
            table[(col, 'Change (vs VAR)')] = np.round((value - table[(col, 'mean')]) / value * 100, 2)
            table[(col, 'Change (vs VAR)')] = table[(col, 'Change (vs VAR)')].astype(str) + '\%'
            # Replace 0\% with -
            table[(col, 'Change (vs VAR)')] = table[(col, 'Change (vs VAR)')].replace('0.0\%', '-')
        table = table.drop(columns=[(col, 'std'), (col, 'mean')])
    return table


def compute_p_values(results, models, counterparts):
    """
    Compute p-values for the difference between models and their counterparts
    :param results: converged results
    :param models: model names
    :param counterparts: list of counterparts for each model
    :return: table of p-values with * if p-value <= 0.05
    """
    p_values = []
    for metric in ['Matrix RMSE', 'Test RMSE', 'Same Sign %']:
        temp = [metric, 'VAR']
        temp2 = [metric, 'True model']
        for i in range(len(models)):
            model = models[i]
            if isinstance(counterparts[i], str):
                temp.append(apply_significance(np.round(stats.ttest_1samp(
                    results[results['Model'] == model][metric].values - results[results['Model'] == counterparts[i]][
                        metric].values, 0)[1], 3), 0.05))
            else:
                temp.append(apply_significance(np.round(stats.ttest_1samp(
                    results[results['Model'] == model][metric].values - results[results['Model'] == counterparts[i][0]][
                        metric].values, 0)[1], 3), 0.05))
                temp2.append(apply_significance(np.round(stats.ttest_1samp(
                    results[results['Model'] == model][metric].values - results[results['Model'] == counterparts[i][1]][
                        metric].values, 0)[1], 3), 0.05))

        p_values.append(pd.DataFrame(temp, index=['p-value of', 'against'] + models).T)
        if len(temp2) > 2:
            p_values.append(pd.DataFrame(temp2, index=['p-value of', 'against'] + models).T)
    return pd.concat(p_values).set_index(['p-value of', 'against']).T[['Matrix RMSE', 'Test RMSE', 'Same Sign %']]
