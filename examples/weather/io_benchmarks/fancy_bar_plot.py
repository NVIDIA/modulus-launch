import matplotlib.pyplot as plt


def fancy_bar_plot(data, labels, title="", ylabel="", xlabel="", save_path=None):
    """
    Create a fancy bar plot using the given parameters.

    Parameters
    ----------
    data : list
        List of data to plot.
    labels : list
        List of labels for each data.
    title : str, optional
        Title of the plot, by default "".
    ylabel : str, optional
        Label for the y-axis, by default "".
    xlabel : str, optional
        Label for the x-axis, by default "".
    save_path : str, optional
        Path to save the plot, by default None.
    """

    # Check if the length of data and labels are the same
    if len(data) != len(labels):
        raise ValueError("Length of data and labels must be the same.")

    # Create the bar plot with customization
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set color for the bars (GPU green and CPU gray)
    colors = []
    for l in labels:
        if "gpu" in l.lower():
            colors.append("#77B900")
        else:
            colors.append("#666666")

    # Bar color and background customization
    bars = ax.bar(labels, data, color=colors, edgecolor="black")
    ax.set_facecolor("white")
    fig.set_facecolor("white")

    # Add title and labels with white text color
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Save the plot if save_path is provided
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, facecolor=fig.get_facecolor())
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
