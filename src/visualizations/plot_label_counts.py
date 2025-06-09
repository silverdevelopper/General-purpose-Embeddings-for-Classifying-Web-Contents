import matplotlib.pyplot as plt
import seaborn as sns

def plot_label_counts(df, label_column='label', figsize=(12, 6), title='Label Distribution', 
                     xlabel='Labels', ylabel='Count', rotation=45, save_path=None):
    """
    Create a bar chart of label counts from a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the labels
    label_column : str, default='label'
        Name of the column containing the labels
    figsize : tuple, default=(12, 6)
        Figure size (width, height)
    title : str, default='Label Distribution'
        Title of the plot
    xlabel : str, default='Labels'
        Label for x-axis
    ylabel : str, default='Count'
        Label for y-axis
    rotation : int, default=45
        Rotation angle for x-axis labels
    save_path : str, optional
        If provided, save the plot to this path
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Count the labels
    label_counts = df[label_column].value_counts()
    
    # Set the style
    plt.style.use('seaborn')
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot
    sns.barplot(x=label_counts.index, y=label_counts.values, ax=ax)
    
    # Customize the plot
    plt.title(title, pad=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha='right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig

# Example usage:
# import pandas as pd
# df = pd.DataFrame(...)
# fig = plot_label_counts(df, save_path='label_distribution.png')
# plt.show()
