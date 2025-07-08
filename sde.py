import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    plt.style.use("default")
    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Assignment
    Write a program to integrate the following stochastic differential equation:

    $$dx = −xdt + \sqrt{2}dW$$
    Use Euler integrator (i.e. just replace $dt$ with $\Delta t$ and $dW$ with $\sqrt{\Delta t}R$
    where R is a Gaussian number with zero average and unitary variance;
    choose a short timestep $\Delta t=0.001$).
    """
    )
    return


@app.cell
def _(mo, np, plt):
    def x_evolution(x, dt):
        return -x * dt + np.sqrt(2 * dt) * np.random.normal(0, 1)

    def simulate_traj(x, total_time, dt, evo, abs=False, seed=69):
        np.random.seed(seed)
        nsteps = int(total_time / dt)
        trajectory = [x]
        for _ in range(nsteps):
            x += evo(x, dt)
            if abs:
                x = np.abs(x)
            trajectory.append(x)
        time = np.linspace(0, total_time, nsteps + 1)
        return (time, trajectory)

    def plot_trajectory(time, trajectory, dt, ax):
        ax.plot(time, trajectory, label=f'dt={dt}')
        ax.set_title(f'Trajectory with dt={dt}')
        ax.set_xlabel('Time')
        ax.set_ylabel('x')
        ax.grid()
        ax.legend()
    dt_values = [0.01, 0.001, 0.0005, 0.0002]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, dt in enumerate(dt_values):
        time, trajectory = simulate_traj(10, 20, dt, x_evolution)
        plot_trajectory(time, trajectory, dt, axes[i])
    mo.center(plt.gca())
    return axes, dt, simulate_traj, time, x_evolution


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Assignment
    Using this program, compute 100 trajectories with same initial condition (x = 10) and different seeds for the random number generator. 
    At fixed value of time t, compute the average over the trajectories of the value of
    x and its standard deviation. How do these two quantities depend on t?
    """
    )
    return


@app.cell
def _(
    axes,
    dt,
    np,
    plot_trajectory_with_stats,
    plt,
    simulate_traj,
    time,
    x_evolution,
):
    _time = 10
    _dt = 10

    trajs = np.empty((100, int(_time / _dt) + 1))
    for j in range(100):
        _, traj = simulate_traj(10, _time, _dt, x_evolution, abs, seed=j)
        trajs[j, :] = traj
  
    mean_trajectory = np.mean(trajs, axis=0)
    variance_trajectory = np.var(trajs, axis=0)

    _timestamps = []
    plt.plot(time, mean_trajectory, label="Mean Trajectory")
    plt.plot(time, variance_trajectory, label="Variance (Line)", linestyle="--")
    plt.set_title("TITOLO")
    plt.set_xlabel("Time")
    plt.set_ylabel("Value")
    plt.legend()
    plt.grid()
    plot_trajectory_with_stats(np.linspace(0, time, int(time / dt) + 1), trajs, axes[0], title=f'Time={time}')

    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Comment
    - Variance seems to stabilize around $sqrt{2}$ which is expected
    - The average has the exponential decay expected from $dx = -x dt$ due to the fact that the noise has average $0$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Assignment
    Use the Ito chain rule to derive the Equation of motion for the variable $y = x^2$ and implement the corresponding algorithm. Run multiple simulations on $x$ or $y$ with equivalent conditions. Notice that you should make sure $y$ stays
    positive while propagating it. A possible option is to reset $y$ to its absolute value at evert step. Do the statistical properties of $x^2$ and $y$ match?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    By Ito chain rule $dy = y' (Adx + Bdw) + \frac{1}{2}y'' B^2 dt$. In this case $A = -1$ and $B = \sqrt{2}$. This results in
    $$
        dy = 2x (-x dt + \sqrt{2} dw) + 2 dt = 2(1-y)dt + 2 \sqrt{2y} dw 
    $$
    """
    )
    return


@app.cell
def _(np):
    def y_evolution(y, dt):
        return 2 * (1 - y) * _dt + np.sqrt(8 * y * _dt) * np.random.normal(0, 1)
    return (y_evolution,)


@app.cell
def _(run_100_trajs, y_evolution):
    _times = [10, 20, 30, 40]
    _dt = 0.001
    run_100_trajs(_times, _dt, y_evolution, True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Assignment
    Implement a code propagating y using a naive chain rule (disregarding the Ito term $\frac{y''}{2}B^2 dt$). Do the statistical property of this new set of trajectories match those obtained before?
    """
    )
    return


@app.cell
def _(np):
    def y_evolution_wrong(y, dt):
        return -2 * y * _dt + np.sqrt(8 * y * _dt) * np.random.normal(0, 1)
    return (y_evolution_wrong,)


@app.cell
def _(run_100_trajs, y_evolution_wrong):
    _times = [10, 20, 30, 40]
    _dt = 0.001
    run_100_trajs(_times, _dt, y_evolution_wrong, True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Comment
    - Trajectories are wrong (obviously)
    - Variance is under estimated
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Assignment

    Implement an implicit integrator for the equation obtained for $y$ at the previous point, where the prefactor for the Wiener noise is computed at the mid point as it is done in the Stratonovich formalism of stochastic differential equations. To do so you should iterate the solution at every step, until convergence is achieved within some numerical threshold. Do the statistical property of this new set of trajectories match those obtained before?
    """
    )
    return


@app.function
def stratonovich():
    # TODO
    pass


if __name__ == "__main__":
    app.run()
