import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def plot_performances_over_epochs(metrics, thresholds, palette, figsize=(11, 7), fontsize=11):
    def plot_epochs(fig, data, column, palette, errorbar=False, threshold=0.1):
        sns.lineplot(data=data, x='Epoch', y=column, hue='Model', errorbar='ci' if errorbar else None, ax=fig,
                     palette=palette)

        lim = data[data['Epoch'] > data.Epoch.max() * threshold].groupby(['Model', 'Epoch']).mean()[column].describe()
        y_low = lim['min']
        y_high = lim['max']
        fig.set_ylim(y_low - lim['std'], y_high + lim['std'])

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    plot_epochs(axs[0][0], metrics, 'Loss', palette=palette, threshold=thresholds[0])
    plot_epochs(axs[0][1], metrics, 'Validation Loss', palette=palette, threshold=thresholds[1])
    plot_epochs(axs[1][1], metrics, 'Matrix RMSE', palette=palette, threshold=thresholds[3])
    plot_epochs(axs[1][0], metrics, 'Test RMSE', palette=palette, threshold=thresholds[2])
    handles, labels = axs[0][0].get_legend_handles_labels()
    axs[0][1].legend(handles, labels, fontsize=fontsize, bbox_to_anchor=(1, 0.5), loc='center left', title='Model')
    axs[0][0].get_legend().remove()
    axs[1][0].get_legend().remove()
    axs[1][1].get_legend().remove()
    plt.tight_layout()
    plt.show()
    return fig


def plot_gradients_over_epochs(gradients, palette, figsize=(10, 10), fontsize=10):
    # TODO rewrite for varying number of variables
    fig, axs = plt.subplots(3, 2, figsize=figsize)
    # For each variable, plot the mean gradient over epochs for each model
    for i in range(len(gradients.Variable.unique())):
        variable_name = gradients.Variable.unique()[i]
        j = 0 if 'LSTM' in variable_name else 1
        i = i % 3
        data = gradients[gradients['Variable'] == variable_name]
        sns.lineplot(data=data, x='Epoch', y='mean', hue='Model', ax=axs[i][j], errorbar='ci', legend=i == 1 and j == 1,
                     palette=palette)
        axs[i][j].set_ylabel(variable_name)
        axs[i][j].set_xlabel('Epoch')
        lim = data.drop(columns=['Variable']).groupby(['Epoch', 'Model']).mean().reset_index()
        lim = lim[lim['Epoch'] > lim['Epoch'].max() * 0.12]['mean'].describe()
        axs[i][j].set_ylim(lim['min'] - lim['std'], lim['max'] + lim['std'])

    axs[1][1].legend(fontsize=fontsize, loc='center left', bbox_to_anchor=(1, 0.5), title='Model')
    plt.tight_layout()
    plt.show()
    return fig


def plot_convergence_differences(to_plot, figsize=(12, 10), fontsize=11):
    def plot_histograms(fig, data, columns, hue, legend=False, colors=None):
        if colors is None:
            colors = sns.color_palette('colorblind', n_colors=len(data[0][hue].unique()))
        sns.histplot(data=data[0], x=columns[0], hue=hue, ax=fig[0], multiple="stack", palette=colors)
        sns.histplot(data=data[1], x=columns[1], hue=hue, ax=fig[1], multiple="stack", palette=colors)
        fig[0].get_legend().remove()
        if legend:
            legend = fig[1].legend(data[0][hue].unique(), fontsize=fontsize, loc='center left', bbox_to_anchor=(1, 0.5),
                                   title='Model')
            for i, legend_entry in enumerate(legend.legend_handles):
                legend_entry.set_color(colors[i])
        else:
            fig[1].get_legend().remove()

    fig, ax = plt.subplots(2, len(to_plot), figsize=figsize, sharey='all')
    for i in range(len(to_plot)):
        plot_histograms(ax[i], to_plot[i]['df'], to_plot[i]['columns'], to_plot[i]['hue'], legend=to_plot[i]['legend'],
                        colors=to_plot[i]['palette'])
    return fig


def plot_heatmap(results, vars, lstm_vars, figsize=(12, 7)):
    r = results.replace(0, np.nan)

    x_columns = [col for col in results.columns if 'x_' in col]
    lag_columns = [col for col in results.columns if 'lag_' in col]
    # Split x_columns and lag_columns into 2 lists each, one where the values end with _sum and one with the rest
    x_columns_sign = [col for col in x_columns if col.endswith('_sign')]
    x_columns = [col for col in x_columns if not col.endswith('_sign')]
    lag_columns_sign = [col for col in lag_columns if col.endswith('_sign')]
    lag_columns = [col for col in lag_columns if not col.endswith('_sign')]

    # Drop nan columns
    r = r.drop(columns=r.columns[r.isna().all()])
    lag_columns = [col for col in r.columns if 'lag_' in col]
    lag_columns_sign1 = [col for col in lag_columns if col.endswith('_sign')]
    lag_columns1 = [col for col in lag_columns if not col.endswith('_sign')]

    fig, ax = plt.subplots(2, figsize=figsize)
    sns.heatmap(
        data=r[['Model'] + x_columns + lag_columns1].groupby('Model').mean().reindex(vars + lstm_vars).replace(0,
                                                                                                               np.nan),
        cmap='Reds', annot=False, fmt='.2f', cbar=True, ax=ax[0])
    ax[0].title.set_text('RMSE')
    plt.setp(ax[0].get_xticklabels(), rotation=90)
    sns.heatmap(data=r[['Model'] + x_columns_sign + lag_columns_sign1].groupby('Model').mean().reindex(
        vars + lstm_vars).replace(0, np.nan), cmap='Blues', annot=False, fmt='.2f', cbar=True, ax=ax[1])
    ax[1].title.set_text('Sign %')
    plt.tight_layout()
    plt.show()
    return fig


def plot_forecast_breakdown(model, dataset, time_series, dgp):
    r = model(dataset.X_train_all, training=False)
    # Subplot with 3 rows
    fig, axs = plt.subplots(2, 3, figsize=(15, 6))
    # Plot forecast vs true and residuals on right

    # Forecast vs true
    axs[0][0].plot(dataset.y_train_all[:, 0, time_series][:500], label='y')
    axs[0][0].plot(r[0][:, time_series][:500], label='r', alpha=0.8)
    axs[0][0].axvline(x=dataset.X_train.shape[0] - 500, c='r', linestyle='--')
    axs[0][0].legend(['True', 'Forecast'])
    axs[0][0].set_title('Forecast vs True')
    # Residuals as histogram
    axs[0][1].hist(r[0][:, time_series][:500] - dataset.y_train_all[:, 0, time_series][:500], label='r-y')
    axs[0][1].set_title('Residuals')
    #axs[0][1].plot(r[0][:, time_series][:500] - dataset.y_train_all[:, 0, time_series][:500], label='r-y')
    #axs[0][1].axvline(x=dataset.X_train.shape[0] - 500, c='r', linestyle='--')
    #axs[0][1].set_title('Residuals')
    # Linear
    axs[1][0].plot(r[1]['linear'][:, time_series][:500], label='linear')
    axs[1][0].axvline(x=dataset.X_train.shape[0] - 500, c='r', linestyle='--')
    axs[1][0].set_title('Linear')
    # non-linear
    axs[1][1].plot(r[1]['non_linear'][:, time_series][:500], label='non-linear')
    axs[1][1].axvline(x=dataset.X_train.shape[0] - 500, c='r', linestyle='--')
    axs[1][1].set_title('Non-linear')
    # Backcasted
    axs[1][2].plot(r[1]['backcasted'][:, 0, time_series][:500], label='backcasted')
    axs[1][2].axvline(x=dataset.X_train.shape[0] - 500, c='r', linestyle='--')
    axs[1][2].set_title('Backcasted Input')
    # Make axs[0][2] empty
    axs[0][2].axis('off')

    # Plot non-linear, linear, and input to linear for vts
    # Title for whole plot
    fig.suptitle(f'Simulation 0, Time Series {time_series}, DGP=' + dgp)
    plt.show()
    return fig