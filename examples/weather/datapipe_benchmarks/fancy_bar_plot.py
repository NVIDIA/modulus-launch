import matplotlib.pyplot as plt

def fancy_bar_plot(data, labels, title="", ylabel="", xlabel="", save_path=None):
    """
    Create a fancy bar plot using the given parameters.
    
    Parameters:
    - data (list): List of numbers representing the height of each bar.
    - labels (list): List of strings representing the label for each bar.
    - title (str): Title for the plot.
    - ylabel (str): Label for the y-axis.
    - xlabel (str): Label for the x-axis.
    - save_path (str, optional): If provided, saves the plot to the given path.
    """
    
    # Check if the length of data and labels are the same
    if len(data) != len(labels):
        raise ValueError("Length of data and labels must be the same.")
    
    # Create the bar plot with customization
    fig, ax = plt.subplots(figsize=(10,6))
   
    # Set color for the bars (GPU green and CPU gray)
    colors = []
    for l in labels:
        if 'gpu' in l.lower():
            colors.append('#77B900')
        else:
            colors.append('#666666')

    # Bar color and background customization
    bars = ax.bar(labels, data, color=colors, edgecolor='white')
    ax.set_facecolor('black')
    fig.set_facecolor('black')
    
    # Add title and labels with white text color
    plt.title(title, color='white')
    plt.xlabel(xlabel, color='white')
    plt.ylabel(ylabel, color='white')
    
    # Customize the axes colors
    ax.tick_params(axis='both', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')

    # Save the plot if save_path is provided
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, facecolor=fig.get_facecolor())
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
