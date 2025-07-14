import marimo

__generated_with = "0.14.10"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Harmonic Oscillator
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pylot as plt 

    # Parameters
    q = np.array([1, 1.1, 0.9, 0.85, 1.15])
    p = np.array([1, 1.1, 0.9, 0.85, 1.15])
    dt = 0.1
    steps = 1000  # Number of integration steps


    # Functions
    def U(q):
        return np.square(q) / 2


    def K(p):
        return np.square(p) / 2


    def force(q):
        return -q.copy()


    # Lists to store the evolution
    q_values = []
    p_values = []

    # Integration loop with energy rescaling
    for _ in range(steps):
        E_initial = U(q) + K(p)
        p += force(q) * dt / 2
        q += p * dt
        p += force(q) * dt / 2

        # Rescale momenta
        E_final = U(q) + K(p)
        rescale_factor = np.sqrt((K(p) + E_initial - E_final) / K(p))
        p *= rescale_factor

        q_values.append(q.copy())
        p_values.append(p.copy())

    q_values = np.array(q_values)
    p_values = np.array(p_values)

    # Visualization: Phase Space
    plt.figure(figsize=(8, 6))
    for i in range(len(q)):  # Loop through each particle
        plt.plot(q_values[:, i], p_values[:, i], label=f"Particle {i+1}")

    plt.title("Phase Space Trajectories (With Energy Rescaling)", fontsize=14)
    plt.xlabel("Position (q)", fontsize=12)
    plt.ylabel("Momentum (p)", fontsize=12)
    plt.legend()
    plt.grid()
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
