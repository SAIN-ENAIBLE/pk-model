import matplotlib.pyplot as plt

# expected input  [t, qc, qp1, qp2, qp3, qp4 ....]


def plot_quantity(data, title=None, xlabel="Time (h)", ylabel="Quantity (ng)"):
    """
    Plot the quantity of drug in each compartment over time.

    Parameters:
    - data: A 2D array where the first column is time and subsequent columns are drug quantities in compartments.
    - title: Title of the plot (optional).
    - xlabel: Label for the x-axis (default is 'Time (h)').
    - ylabel: Label for the y-axis (default is 'Quantity (ng)').
    """
    time = data[:, 0]
    num_compartments = data.shape[1] - 1

    plt.figure(figsize=(10, 6))

    for i in range(1, num_compartments + 1):
        plt.plot(time, data[:, i], label=f"Compartment {i}")

    plt.title(title if title else "Drug Quantity Over Time")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()
