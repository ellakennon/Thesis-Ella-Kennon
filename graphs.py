import seaborn as sns            
import matplotlib.pyplot as plt
import numpy as np 

from helpers import gen_sample_vars

# Graph MAE data for given n, k
def graph_multibasis_mae(dfs, labels, colors, n, k, subtitle, problem_name):

    # Function to plot dfs individually
    def _plot(df, label, color):
        # Check that df is passed
        if df is None:
            return
        
        # Get relevant data for n, k, sorted by size
        rel_df = (df[(df['n'] == n) & (df['k'] == k)])

        # Plot
        sns.lineplot(
            data=rel_df,
            x='sample_size',
            y='mae',
            estimator='mean',
            errorbar=('ci', 95),
            marker='o',
            label=label,
            color=color
        )
    
    # Use helper function to make each plot
    for df, label, color in zip(dfs, labels, colors):
        _plot(df, label, color)

    plt.xlabel('Number of Training Samples')
    plt.ylabel('Average MAE')
    plt.title(f'Average MAE over Training Sizes\n{subtitle}\n{problem_name} (n={n}, k={k})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot optimization data from two dataframes
def plot_optimization(dfs, labels, colors, n, k, subtitle, problem_name):
    # Function to plot dfs individually
    def _plot(df, label, color):    
        if df is None:
            return
            
        # Plot
        plot_df = df[['sample_size', 'predicted_best']]

        sns.lineplot(
            data=plot_df,
            x='sample_size',
            y='predicted_best',
            estimator='mean',
            errorbar=('ci', 95),
            marker='o',
            label=label,
            color=color
        )
    
    for df, label, color in zip(dfs, labels, colors):
        _plot(df, label, color)
    
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Fitness')
    plt.title(f'Hamming-Ball Efficient Hill Climber Optimization\n{subtitle}\n{problem_name} (n={n}, k={k})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()