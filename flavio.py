import marimo

__generated_with = "0.14.10"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    def _sample_shifted_exponential(size=1, random_seed=None):
        """
        Samples from the shifted exponential distribution with mean 0 and variance 1.

        Parameters:
            size (int): Number of samples to generate.
            random_seed (int, optional): Random seed for reproducibility.

        Returns:
            numpy.ndarray: Samples from the distribution.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        exp_samples = np.random.exponential(scale=1.0, size=size)
        shifted_samples = np.sqrt(2) * (exp_samples - 1)
        return shifted_samples

    def _plot_histogram(samples, bins=50):
        """
        Plots a histogram of the given samples.

        Parameters:
            samples (numpy.ndarray): Samples to plot.
            bins (int): Number of bins for the histogram.
        """
        plt.figure(figsize=(8, 5))
        plt.hist(_samples, bins=bins, density=True, alpha=0.7, color='blue', label='Samples')
        plt.title('Histogram of Shifted Exponential Distribution')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid()
        plt.show()
    if __name__ == '__main__':
        _samples = _sample_shifted_exponential(size=10000, random_seed=42)
        _plot_histogram(_samples)
    return np, plt


@app.cell
def _(np, plt):
    def _sample_shifted_exponential(size=1, random_seed=None):
        """
        Samples from the shifted exponential distribution with mean 0 and variance 1.

        Parameters:
            size (int): Number of samples to generate.
            random_seed (int, optional): Random seed for reproducibility.

        Returns:
            numpy.ndarray: Samples from the distribution.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        exp_samples = np.random.exponential(scale=1.0, size=size)
        shifted_samples = np.sqrt(2) * (exp_samples - 1)
        return shifted_samples

    def _plot_histogram(samples, bins=50):
        """
        Plots a histogram of the given samples.

        Parameters:
            samples (numpy.ndarray): Samples to plot.
            bins (int): Number of bins for the histogram.
        """
        plt.figure(figsize=(8, 5))
        plt.hist(_samples, bins=bins, density=True, alpha=0.7, color='blue', label='Samples')
        plt.title('Histogram of Shifted Exponential Distribution')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid()
        plt.show()

    def integrate_dx(initial_x, dt, t_final, random_seed=None):
        """
        Integrates the stochastic differential equation dx = -x dt + dw,
        where dw are samples from the shifted exponential distribution.

        Parameters:
            initial_x (float): Initial value of x.
            dt (float): Time step for the integration.
            t_final (float): Final time for the integration.
            random_seed (int, optional): Random seed for reproducibility.

        Returns:
            numpy.ndarray: Time array.
            numpy.ndarray: Integrated values of x over time.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        num_steps = int(t_final / dt)
        t = np.linspace(0, t_final, num_steps + 1)
        x = np.zeros(num_steps + 1)
        x[0] = initial_x
        for i in range(1, num_steps + 1):
            dw = _sample_shifted_exponential(size=1)[0] * np.sqrt(dt)
            x[i] = x[i - 1] - x[i - 1] * dt + dw
        return (t, x)
    if __name__ == '__main__':
        _samples = _sample_shifted_exponential(size=10000, random_seed=42)
        _plot_histogram(_samples)
        t, x = integrate_dx(initial_x=1.0, dt=0.01, t_final=10.0, random_seed=42)
        plt.figure(figsize=(8, 5))
        plt.plot(t, x, label='Integrated x(t)')
        plt.title('Integration of dx = -x dt + dw')
        plt.xlabel('Time t')
        plt.ylabel('x(t)')
        plt.legend()
        plt.grid()
        plt.show()
    return


if __name__ == "__main__":
    app.run()
