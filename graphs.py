import seaborn as sns            
import matplotlib.pyplot as plt 

# Graph MAE data for given n, k
def graph_multibasis_mae(poly_df=None, lasso_df=None, lars_df=None, ll_df=None, n=None, k=None, problem_name=None, basis=None, sample_size_min=None):

    # Function to plot dfs individually
    def _plot(df, label, color):
        # Check that df is passed
        if df is None:
            return
        
        # Get relevant data
        rel_df = df[(df['n'] == n) & (df['k'] == k)]

        # Removes smaller sample sizes if needed
        if sample_size_min is not None:
            rel_df = rel_df[rel_df['sample_size'] >= sample_size_min]

        # Plot
        sns.lineplot(
            data=rel_df,
            x='sample_size',
            y='mae',
            estimator='mean',
            errorbar=("ci", 95),
            marker='o',
            label=label,
            color=color
        )
    
    # Use helper function to make each plot
    _plot(poly_df, 'Polynomial Regression', 'blue')
    _plot(lasso_df, 'Lasso Regression', 'green')
    _plot(lars_df, 'Least Angle Regression', 'red')
    _plot(ll_df, 'LassoLars Regression', 'yellow')

    plt.xlabel("Number of Training Samples")
    plt.ylabel("Average MAE")
    plt.title(f"Average MAE vs Training Sample Sizes\n{problem_name}\n{basis} Basis (n={n}, k={k})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot optimization data from two dataframes
def plot_optimization(df1, df2, label1, label2, n, k, problem_name, basis):
    # Function to plot dfs individually
    def _plot(df, label, color):        
        # Plot
        plot_df = df[['sample_size', 'predicted_best']]

        sns.lineplot(
            data=plot_df,
            x='sample_size',
            y='predicted_best',
            estimator='mean',
            errorbar=("ci", 95),
            marker='o',
            label=label,
            color=color
        )

    _plot(df1, label=label1, color='blue')
    _plot(df2, label=label2, color='red')
    
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Fitness")
    plt.title(f"Hamming-Ball Efficient Hill Climber Optimization\n{problem_name}\n{basis} Basis (n={n}, k={k})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()