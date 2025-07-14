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
    The Verlet algorithms derives directly from the use of Taylor expansion on the Newton equation:
    $$q(t+\Delta t) = q(t) + \dot{q}(t) \Delta t + \frac{\ddot{q}(t)}{2}\Delta t^2 + \frac{\dddot{q}(t)}{3!}\Delta t^3 + O(\Delta t^4)$$
    $$q(t-\Delta t) = q(t) - \dot{q}(t) \Delta t + \frac{\ddot{q}(t)}{2}\Delta t^2 - \frac{\dddot{q}(t)}{3!}\Delta t^3 + O(\Delta t^4)$$
    $$q(t+\Delta t)  + q(t-\Delta t) = 2q(t) + \ddot{q}(t)\Delta t^2 + O(\Delta t^4)$$
    Thus resulting in the following update function:
    $$q(t+\Delta t) = 2q(t) - q(t-\Delta t) + \frac{f(t)}{m} \Delta t^2$$

    The numerical solutions obtained with this algorithm are **time reversible** but **don't conserve energy**. Note that we don't have the velocity playing a role in the update, this implies that the initial conditions have to be given as a tuple of 2 positions, intead of a tuple of position and velocity


    ## Code
    ```python
    def verlet(
        q_curr,
        q_prev,
        mass,
        dt,
        n_steps,
        stride,
        force,
        force_args,
    ):
        q_list = []
        times = []

        for i in range(0, n_steps):
            if (i + 1) % stride == 0:
                q_list.append(q_curr)
                times.append((i + 1) * dt)

            q_next = (
                2 * q_curr - q_prev + force(q_curr, *force_args) / mass * dt**2
            )
            q_prev = q_curr
            q_curr = q_next

        return (times, q_list)
    ```
    """
    )
    return


@app.function(hide_code=True)
def verlet(
    q_curr,
    q_prev,
    mass,
    dt,
    n_steps,
    stride,
    force,
    force_args,
):
    q_list = []
    times = []

    for i in range(0, n_steps):
        if (i + 1) % stride == 0:
            q_list.append(q_curr)
            times.append((i + 1) * dt)

        q_next = (
            2 * q_curr - q_prev + force(q_curr, *force_args) / mass * dt**2
        )
        q_prev = q_curr
        q_curr = q_next

    return (times, q_list)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Velocity Verlet
    Applying the Liouville formalism we can proceed by approximate the evolution operator by means of **Trotter splitting**: 
    $$\exp\left[{-\Delta t \left(\hat{L}_q + \hat{L}_p\right)}\right] = \exp\left[-\frac{\Delta t}{2}\hat{L}_p\right]\exp\left[-\Delta t\hat{L}_q\right]\exp\left[-\frac{\Delta t}{2}\hat{L}_p\right] + o(\Delta t^3)$$

    By doing this we get an algorithm that is **time reversible** and **preserve incrompressibility of the flow**, unfortunately it **does not conserve energy** since at each substep we use a different Hamiltonian

    An implementation of this code would be (as usual we update the force when the position is updated): 
    ```python
    def velocity_verlet(
        q,
        p,
        m,
        dt,
        steps,
        stride,
        force,
        force_args,
    ):
        q_list = []
        p_list = []
        f = force(q, *force_args)
        for i in range(steps):
            if (i + 1) % stride == 0:
                q_list.append(q)
                p_list.append(p)
            p += f * dt / 2
            q += p * dt / m
            f = force(q, *force_args)
            p += f * dt / 2
        return (q_list, p_list)
    ```
    """
    )
    return


@app.function(hide_code=True)
def velocity_verlet(
    q,
    p,
    m,
    dt,
    steps,
    stride,
    force,
    force_args,
):
    q_list = []
    p_list = []
    f = force(q, *force_args)
    for i in range(steps):
        if (i + 1) % stride == 0:
            q_list.append(q)
            p_list.append(p)
        p += f * dt / 2
        q += p * dt / m
        f = force(q, *force_args)
        p += f * dt / 2
    return (q_list, p_list)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Position Verlet
    Applying the Liouville formalism we can proceed by approximate the evolution operator by means of **Trotter splitting**: 
    $$\exp\left[{-\Delta t \left(\hat{L}_q + \hat{L}_p\right)}\right] = \exp\left[-\frac{\Delta t}{2}\hat{L}_q\right]\exp\left[-\Delta t\hat{L}_p\right]\exp\left[-\frac{\Delta t}{2}\hat{L}_q\right] + o(\Delta t^3)$$

    By doing this we get an algorithm that is **time reversible** and **preserve incrompressibility of the flow**, unfortunately it **does not conserve energy** since at each substep we use a different Hamiltonian. 

    An implementation of this code would be (as usual we update the force when the position is updated): 
    ```python
    def position_verlet(
        q,
        p,
        m,
        dt,
        steps,
        stride,
        force,
        force_args,
    ):
        q_list = []
        p_list = []
        f = force(q, *force_args)
        for i in range(steps):
            if (i + 1) % stride == 0:
                q_list.append(q)
                p_list.append(p)
            q = q + p/m * dt /2
            f = force(q)
            p = p + f * dt
            q = q + p/m + dt/2
        return (q_list, p_list)
    ```
    """
    )
    return


@app.function(hide_code=True)
def position_verlet(
    q,
    p,
    m,
    dt,
    steps,
    stride,
    force,
    force_args,
):
    q_list = []
    p_list = []
    f = force(q, *force_args)
    for i in range(steps):
        if (i + 1) % stride == 0:
            q_list.append(q)
            p_list.append(p)
        q = q + p/m * dt /2
        f = force(q)
        p = p + f * dt
        q = q + p/m + dt/2
    return (q_list, p_list)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Application
    Here there's an interactive environment to play around with different potentials and integration algorithms
    """
    )
    return


@app.function(hide_code=True)
def hooke(q, k=1):
    return -1 * k * q


@app.function(hide_code=True)
def lj(q, epsilon=1, sigma=1):
    return 48 * epsilon * (sigma**12 / (q**13) - 0.5 * (sigma**6) / (q**7))


@app.cell(hide_code=True)
def _(np, param_1, param_2, param_3):
    def U_lj(r, e=param_1.value, s=param_2.value):
        return 4 * e * ((s / r) ** 12 - (s / r) ** 6)


    def U_morse(r, D=param_1.value, c=param_2.value, a=param_3.value):
        return D*(1 - np.exp(-a*(r - c))) ** 2


    def morse(r, D=param_1.value, c=param_2.value, a=param_3.value):
        return -2 * a * D * (1 - np.exp(-a * (r - c))) * np.exp(-a*(r - c))
    return U_lj, U_morse, morse


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Simulation""")
    return


@app.cell(hide_code=True)
def _(mo):
    algorithm = mo.ui.dropdown(
        value="Verlet",
        options=["Verlet", "Velocity Verlet", "Position Verlet"],
    )
    potential = mo.ui.dropdown(
        value="Hooke",
        options=["Hooke", "Lennard-Jones", "Morse"],
    )
    timestep = mo.ui.number(
        value=0.4, step=0.01, start=0
    )
    mass = mo.ui.number(value=1, step=0.5)
    simulation_time = mo.ui.number(value=100, step=10)
    stride = mo.ui.number(value=1, step=1)

    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md("Algorithm: "),
                    mo.md("Potential: "),
                    mo.md("Timestep $(\Delta t)$"),
                    mo.md("Mass $(m)$"),
                    mo.md("Simulation Time"),
                    mo.md("Stride"),
                ]
            ),
            mo.vstack(
                [algorithm, potential, timestep, mass, simulation_time, stride]
            ),
        ],
        widths=[None,1]
    )
    return algorithm, mass, potential, simulation_time, stride, timestep


@app.cell(hide_code=True)
def _(algorithm, mo, potential):
    initial_coord_1 = mo.ui.number(value=2, step=0.1)
    initial_coord_2 = mo.ui.number(value=2, step=0.1)

    labels = {
        "Hooke": ["Elastic constant ($k$): ", "", ""],
        "Lennard-Jones": [
            "Depth ($\epsilon$)",
            "Zero energy point ($\sigma$)",
            "",
        ],
        "Morse": [
            r"Well depth $\left(D_e\right)$",
            "Equilibrium bond distance ($r_c$)",
            r"Well width $(a)$",
        ],
    }

    param_1 = mo.ui.number(value=1, step=0.1)
    param_2 = mo.ui.number(value=1, step=0.1)
    param_3 = mo.ui.number(value=1, step=0.1)

    params = {
        "Hooke": [param_1],
        "Lennard-Jones": [param_1, param_2],
        "Morse": [param_1, param_2, param_3],
    }

    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md("---"),
                    mo.md("$q(t=0)$"),
                    mo.md(
                        "$q(t=-\Delta t)$"
                        if algorithm.value == "Verlet"
                        else "$p(t=0)$"
                    ),
                    mo.md("---"),
                    *[mo.md(v) for v in labels[potential.value]],
                ]
            ),
            mo.vstack(
                [
                    mo.md("---"),
                    initial_coord_1,
                    initial_coord_2,
                    mo.md("---"),
                    *params[potential.value],
                ]
            ),
        ],
        widths=[None, 1],
    )
    return initial_coord_1, initial_coord_2, param_1, param_2, param_3


@app.cell(hide_code=True)
def _(U_lj, U_morse, np, param_1, param_2, param_3, plt):
    def plot_lj(epsilon=param_1.value, sigma=param_2.value):
        r_min = 2 ** (1 / 6) * sigma

        # minimum line
        if r_min != 0:
            # 0 energy point
            plt.axvline(sigma, color="green", lw=0.5, ls="--")
            plt.axvline(r_min, color="red", lw=0.5, ls="--")
            plt.axhline(U_lj(r_min), color="coral", lw=0.5, ls="--")

        # colored labels in top-right corner (axes coords)
        plt.text(
            0.95,
            0.95,
            rf"$q_0 \ = \ \sigma \ = \ {sigma:.2f}$",
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
        # V_min label in coral
        plt.text(
            0.95,
            0.80,
            rf"$V_{{\min}} \ = \ {-epsilon:.2f}$",
            color="coral",
            ha="right",
            va="top",
            transform=plt.gca().transAxes,
        )

        # Plot from 0.9 sigma to see the curve bottom and it's needed a minimum if
        # 0.01 to avoid the singularity in q=0
        start = sigma * 0.9 if abs(sigma * 0.9) > 0.01 else 0.01
        positions = np.linspace(start, 3 * start, 1000)
        plt.title("Lennard-Jones Potential")
        plt.xlabel("q")
        plt.ylabel("U(q)")
        plt.plot(positions, [U_lj(r) for r in positions])


    def plot_hooke(k=param_1.value):
        positions = np.linspace(-3, 3, 1000)
        plt.title("Hooke Potential")
        plt.xlabel("q")
        plt.ylabel("U(q)")
        plt.plot(positions, [k / 2 * r**2 for r in positions])


    def plot_morse(c=param_2.value, a=param_3.value):
        positions = np.linspace(c - a * 0.8, c + 2, 1000)

        plt.axvline(c, color="coral", lw=0.5, ls="--")
        plt.text(
            0.95,
            0.95,
            rf"$V_{{\min}} \ = \ {U_morse(c):.2f}$",
            color="coral",
            ha="right",
            va="top",
            transform=plt.gca().transAxes,
        )

        plt.axhline(U_morse(c), color="red", lw=0.5, ls="--")
        plt.text(
            0.95,
            0.875,
            rf"$q_{{\min}} \  = \ {c:.2f}$",
            color="red",
            ha="right",
            va="top",
            transform=plt.gca().transAxes,
        )
        plt.title("Morse Potential")
        plt.xlabel("q")
        plt.ylabel("U(q)")
        plt.plot(positions, U_morse(positions))
    return plot_hooke, plot_lj, plot_morse


@app.cell(hide_code=True)
def _(
    algorithm,
    initial_coord_1,
    initial_coord_2,
    mass,
    mo,
    morse,
    np,
    param_1,
    param_2,
    param_3,
    plot_hooke,
    plot_lj,
    plot_morse,
    plt,
    potential,
    simulation_time,
    stride,
    timestep,
):
    force_args = {
        "Hooke": (param_1.value,),
        "Lennard-Jones": (param_1.value, param_2.value),
        "Morse": (param_1.value, param_2.value, param_3.value),
    }
    force = {"Hooke": hooke, "Lennard-Jones": lj, "Morse": morse}

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 5))

    _ax1.set(
        xlabel="t",
        ylabel="q(t)",
        title=f"Motion in {potential.value} potential with {algorithm.value}",
    )

    _times = np.linspace(
        0,
        simulation_time.value,
        int(int(simulation_time.value / timestep.value) / stride.value),
    )
    options = (
        initial_coord_1.value,
        initial_coord_2.value,
        mass.value,
        timestep.value,
        int(simulation_time.value / timestep.value),
        stride.value,
        force[potential.value],
        force_args[potential.value],
    )


    match algorithm.value:
        case "Verlet":
            _ax1.plot(
                *verlet(*options),
                label="Verlet",
            )
        case "Velocity Verlet":
            _ax1.plot(
                _times,
                velocity_verlet(*options)[0],
                label="Velocity Verlet",
            )
        case "Position Verlet":
            _ax1.plot(
                _times,
                position_verlet(*options)[0],
                label="Position Verlet",
            )


    if potential.value == "Hooke" and algorithm.value == "Verlet":
        _omega = np.sqrt(param_1.value / mass.value)
        _times = np.linspace(0,simulation_time.value, 3000)
        _ax1.plot(
            _times,
            (
                initial_coord_1.value * np.sin(_omega * (_times + timestep.value))
                - initial_coord_2.value * np.sin(_omega * _times)
            )
            / np.sin(_omega * timestep.value),
            label="Theoretical",
        )
        _ax1.legend(loc="upper right")


    match potential.value:
        case "Hooke":
            plot_hooke()
        case "Lennard-Jones":
            plot_lj()
        case "Morse":
            plot_morse()


    mo.center(plt.gca())
    return


@app.cell(hide_code=True)
def _(mo, potential):
    def info():
        match potential.value:
            case "Hooke":
                return mo.md(r"""
                    ## Hooke Potential
                    Hooke's Law describes the restoring force of an ideal spring, stating that the force exerted by a spring is directly proportional to its displacement from equilibrium. It is fundamental in the study of simple harmonic motion.

                    **Potential Energy:**
                    $$U_H(q) = \frac{1}{2} k q^2$$
                    Where $k$ is the spring constant and $q$ is the displacement from the equilibrium position.

                    **Force:**
                    The force is the negative gradient of the potential energy:
                    $$F_H(q) = -\frac{\partial U_H(q)}{\partial q} = -kq$$
    
                    ```python
                    def hooke(q, k=1):
                        return -1 * k * q
                
                    ```
                    ### Theoretical Solution For Hooke (Verlet Initial Conditions)
                    For a system governed by the Hooke potential, the equation of motion is that of a simple harmonic oscillator. Since the initial conditions provided are on two positions ($q_{prev}$ and $q_{curr}$) rather than position and velocity, the analytical solution for the position $q(t)$ at time $t$ to be confronted with the one given by the algorithm is:
                    $$q(t) = \frac{q_{prev} \sin(\omega (\Delta t - t)) + q_{curr} \sin(\omega t)}{\sin(\omega \Delta t)}$$
                    where:
                
                    - $\omega = \sqrt{k/m}$ is the angular frequency of oscillation (assuming mass $m=1$ if not specified).
                    - $\Delta t$ is the time step between $q_{prev}$ and $q_{curr}$.
                    This form of the solution handles the specified two-position initial conditions.
                """)

            case "Lennard-Jones":
                return mo.md(
                    r"""
                match 
                ## Lennard-Jones Potential
                $$U_{LJ}(q) = 4 \epsilon \left[ \left(\frac{\sigma}{q}\right)^{12} - \left(\frac{\sigma}{q}\right)^6\right]$$
                thus, by computing the gradient we get an expression for the force:
            
                $$F_{LJ}(q) = -\frac{\partial U_{LJ}(q)}{\partial q} = 48\epsilon \left( \frac{\sigma^{12}}{q^{13}}- \frac{\sigma^6}{2q^7}\right)$$
            
                ```python
                def lj(q, epsilon=1, sigma=1):
                    return 48 * epsilon * (sigma**12 / (q**13) - 0.5 * (sigma**6) / (q**7))
                ```
                """
                )
            case "Morse":
                return mo.md(
                    r"""
                    ## Morse Potential
                    The Morse potential describes the potential energy of a diatomic molecule and is commonly used to model interatomic interactions.
                    $$U_M(q) = D_e (1 - e^{-a(q - q_{eq})})^2$$
                
                    where:
                
                    - $D_e$ is the dissociation energy.
                    - $a$ controls the width of the potential well.
                    - $q_{eq}$ is the equilibrium bond distance.

                    The force is derived from the negative gradient of the potential energy:
                    $$F_M(q) = -\frac{\partial U_M(q)}{\partial q} = -2 a D_e (1 - e^{-a(q - q_{eq})}) e^{-a(q - q_{eq})}$$
                
                    ```python
                    def morse(q, D_e=1, a=1, q_eq=0):
                        exponent_term = np.exp(-a * (q - q_eq))
                        return -2 * a * D_e * (1 - exponent_term) * exponent_term
                    ```
                    """
                )


    info()
    return


if __name__ == "__main__":
    app.run()
