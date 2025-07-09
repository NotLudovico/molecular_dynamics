import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use("default")
    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Verlet Algorithm
    The Verlet algorithm updates the coordinates as follows:
    $$q(t+\Delta t) = 2q(t) - q(t-\Delta t) + \frac{f(t)}{m} \Delta t^2$$

    The numerical solutions obtained with this algorithm are:

    - Time reversible
    - Not conserving energy


    ## Code
    ```python
    def verlet(
        *,
        q_curr,
        q_prev,
        mass,
        dt,
        n_steps,
        stride=1,
        force=hooke,
        force_args=(),
    ):
        q_list = []
        times = []

        for i in range(0, n_steps):
            q_next = (
                2 * q_curr - q_prev + force(q_curr, *force_args) / mass * dt**2
            )
            q_prev = q_curr
            q_curr = q_next

            if i % stride == 0:
                q_list.append(q_curr)
                times.append((i + 1) * dt)

        return (times, q_list)
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _():
    def verlet(
        *,
        q_curr,
        q_prev,
        mass,
        dt,
        n_steps,
        stride=1,
        force=hooke,
        force_args=(),
    ):
        q_list = []
        times = []

        for i in range(0, n_steps):
            q_next = (
                2 * q_curr - q_prev + force(q_curr, *force_args) / mass * dt**2
            )
            q_prev = q_curr
            q_curr = q_next

            if i % stride == 0:
                q_list.append(q_curr)
                times.append((i + 1) * dt)

        return (times, q_list)
    return (verlet,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Harmonic Oscillator Application
    In this chapter we apply the algorithm to the hamrmonic oscillator and to the  
    ## Theoretical Solution For Hooke
    Since the initial conditions are on two positions and not on position and velocity as we are used to, the analytical solution to be confronted with the one given by the algorithm is:
    $$q(t) = \frac{q_{prev} \sin(\omega \Delta t) + q_{curr} \sin(\omega t)}{\sin(\omega \Delta t)}$$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Potentials

    ### Hooke Potential
    $$F_H(q) = -kq$$

    ```python
    def hooke(q, k=1):
        return -1 * k * q

    ```
    """
    )
    return


@app.function(hide_code=True)
def hooke(q, k=1):
    return -1 * k * q


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Lennard-Jones Potential
    $$U_{LJ}(q) = 4 \epsilon \left[ \left(\frac{\sigma}{q}\right)^{12} - \left(\frac{\sigma}{q}\right)^6\right]$$
    thus, by computing the gradient we get an expression for the force:

    $$F_{LJ}(q) = -\frac{\partial U_{LJ}(q)}{\partial q} = 48\epsilon \left( \frac{\sigma^{12}}{q^{13}}- \frac{\sigma^6}{2q^7}\right)$$

    ```python
    def lj(q, epsilon=1, sigma=1):
        return 48 * epsilon * (sigma**12 / (q**13) - 0.5 * (sigma**6) / (q**7))
    ```
    """
    )
    return


@app.function(hide_code=True)
def lj(q, epsilon=1, sigma=1):
    return 48 * epsilon * (sigma**12 / (q**13) - 0.5 * (sigma**6) / (q**7))


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Simulation""")
    return


@app.cell(hide_code=True)
def _(mo):
    q_curr = mo.ui.number(value=2, step=0.01)
    q_prev = mo.ui.number(value=2, step=0.01)

    mass = mo.ui.number(value=1, start=0.1, step=0.1)
    hooke_dt = mo.ui.number(value=0.4, step=0.01)
    hooke_k = mo.ui.number(value=1, start=0)
    theoretical_hooke = mo.ui.switch()

    lj_dt = mo.ui.number(value=0.05, step=0.001)
    lj_e = mo.ui.number(value=1, step=0.01)
    lj_sigma = mo.ui.number(value=1, step=0.01)

    stride = mo.ui.number(value=1, start=1, step=1)
    simulation_time = mo.ui.number(value=40)

    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md(r"Initial Position ($q_{curr}$)"),
                    mo.md(r"Previous Position ($q_{prev}$)"),
                    mo.md("---"),
                    mo.md("Simulation Time"),
                    mo.md("Stride"),
                    mo.md("---"),
                    mo.md(r"Lennard-Jones timestep ($\Delta t$)"),
                    mo.md(r"Depth ($\epsilon$)"),
                    mo.md(r"Zero Energy point ($\sigma$)"),
                    mo.md("---"),
                    mo.md(r"Mass ($m$)"),
                    mo.md(r"Hooke Timestep ($\Delta t$)"),
                    mo.md(r"Elastic Constant ($k$)"),
                    mo.md("Show theoretical solution"),
                ],
            ),
            mo.vstack(
                [
                    q_curr,
                    q_prev,
                    mo.md(r"---"),
                    simulation_time,
                    stride,
                    mo.md(r"---"),
                    lj_dt,
                    lj_e,
                    lj_sigma,
                    mo.md(r"---"),
                    mass,
                    hooke_dt,
                    hooke_k,
                    theoretical_hooke,
                ]
            ),
        ],
        widths=[None, 1],
    )
    return (
        hooke_dt,
        hooke_k,
        lj_dt,
        lj_e,
        lj_sigma,
        mass,
        q_curr,
        q_prev,
        simulation_time,
        stride,
        theoretical_hooke,
    )


@app.cell(hide_code=True)
def _(lj_e, lj_sigma):
    def U_lj(r, e=lj_e.value, s=lj_sigma.value):
        return 4 * e * ((s / r) ** 12 - (s / r) ** 6)


    r_min = 2 ** (1 / 6) * lj_sigma.value
    return U_lj, r_min


@app.cell(hide_code=True)
def _(
    hooke_dt,
    hooke_k,
    lj_dt,
    lj_e,
    lj_sigma,
    mass,
    mo,
    np,
    plt,
    q_curr,
    q_prev,
    simulation_time,
    stride,
    theoretical_hooke,
    verlet,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.set(xlabel="t", ylabel="q(t)", title="Motion in Hooke potential")
    ax1.plot(
        *verlet(
            q_curr=q_curr.value,
            q_prev=q_prev.value,
            mass=mass.value,
            dt=hooke_dt.value,
            n_steps=int(simulation_time.value / hooke_dt.value),
            stride=stride.value,
            force=hooke,
            force_args=(hooke_k.value,),
        ),
        label="Verlet",
    )

    if theoretical_hooke.value:
        times = np.linspace(0, simulation_time.value, 1000)
        omega = np.sqrt(hooke_k.value / mass.value)
        ax1.plot(
            times,
            (
                q_curr.value * np.sin(omega * (times + hooke_dt.value))
                - q_prev.value * np.sin(omega * times)
            )
            / np.sin(omega * hooke_dt.value),
            label="Theoretical",
        )
        ax1.legend(loc="upper right")

    ax2.set(xlabel="t", ylabel="q(t)", title="Motion in Lennard-Jones potential")
    ax2.plot(
        *verlet(
            q_curr=q_curr.value,
            q_prev=q_prev.value,
            mass=mass.value,
            dt=lj_dt.value,
            n_steps=int(simulation_time.value / lj_dt.value),
            stride=stride.value,
            force=lj,
            force_args=(lj_e.value, lj_sigma.value),
        )
    )

    mo.center(plt.gca())
    return


@app.cell(hide_code=True)
def _(lj_e, lj_sigma, mo, r_min):
    mo.vstack(
        [
            mo.md("### Minimum"),
            mo.md(
                rf" $q_{{min}} = 2^{{1/6}}  \sigma =  {r_min:.2f}  \quad \Rightarrow \quad V(q_{{min}}) = -\epsilon = {-lj_e.value}$"
            ),
            mo.md("### Zero Energy Point"),
            mo.md(
                rf"$q_0 = \sigma = {lj_sigma.value} \quad \Rightarrow \quad V(q_0)=0$ "
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(U_lj, lj_e, lj_sigma, mo, np, plt, r_min):
    # plot range from 0.01 to 3×minimum to avoid the singularity
    # in r=0 and see the curve approaching 0
    r = np.linspace(0.01, 3 * r_min, 500)

    # plot
    plt.figure(figsize=(6, 4))
    plt.plot(r, [U_lj(i) for i in r], lw=2)

    # Avoid to explode y scale, lower limit just makes
    # the bottom of the potential midway between the x-axes
    # and the end of the plot. Upperlimit is arbitrary,
    # nothing special about that 0.8
    ymin = -lj_e.value * 2
    plt.ylim(ymin, U_lj(r_min * 0.8))
    plt.xlabel(r"$q$")
    plt.ylabel(r"$V(q)$")
    plt.title("Lennard–Jones Potential")

    # x axis
    plt.axhline(0, color="black", lw=0.5)

    # 0 energy point
    plt.axvline(lj_sigma.value, color="green", lw=0.5, ls="--")

    # minimum line
    plt.axvline(r_min, color="red", lw=0.5, ls="--")
    plt.axhline(U_lj(r_min), color="coral", lw=0.5, ls="--")

    # colored labels in top-right corner (axes coords)
    plt.text(
        0.95,
        0.95,
        rf"$q_0 \ = \ \sigma \ = \ {lj_sigma.value:.2f}$",
        color="green",
        ha="right",
        va="top",
        transform=plt.gca().transAxes,
    )
    # r_min label in red
    plt.text(
        0.95,
        0.875,
        rf"$q_{{\min}} \ = \  \sqrt{[6]}{{2}} \ \sigma \ = \ {r_min:.2f}$",
        color="red",
        ha="right",
        va="top",
        transform=plt.gca().transAxes,
    )
    # V_min label in red
    plt.text(
        0.95,
        0.80,
        rf"$V_{{\min}} \ = \ {-lj_e.value:.2f}$",
        color="coral",
        ha="right",
        va="top",
        transform=plt.gca().transAxes,
    )

    mo.center(plt.gca())
    return


if __name__ == "__main__":
    app.run()
