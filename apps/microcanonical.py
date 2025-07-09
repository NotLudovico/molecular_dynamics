import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    plt.style.use("default")
    return math, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    N1 = mo.ui.number(value=6, label="$N_1$")
    N2 = mo.ui.number(value=18, label="$N_2$")
    E = mo.ui.number(value=16, label="$E$")

    mo.vstack([N1, N2, E])
    return E, N1, N2


@app.cell(hide_code=True)
def _(E, N1, N2, math, mo, np):
    # Energy values for N1
    i_vals = np.arange(0, N1.value + 1)

    # Number of cases with index representing number of particles
    success = np.array(
        [
            math.comb(N1.value, i) * math.comb(N2.value, E.value - i)
            if 0 <= E.value - i <= N2.value
            else 0
            for i in i_vals
        ]
    )

    # Statistics
    total = success.sum()
    probs = success / total
    mean_energy = (i_vals * probs).sum()
    second_moment = (i_vals**2 * probs).sum()
    variance = second_moment - mean_energy**2
    std = math.sqrt(abs(variance))

    # Textual output
    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md(f"**Total configurations**: "),
                    mo.md(f"**Mean energy**: "),
                    mo.md(f"**Variance**: "),
                    mo.md(f"**Standard deviation**: "),
                ]
            ),
            mo.vstack(
                [
                    mo.md(f"${int(math.comb(N1.value + N2.value, E.value))}$"),
                    mo.md(f"${mean_energy:.4f}$"),
                    mo.md(f"${variance:.4f}$"),
                    mo.md(f"${std:.4f}$"),
                ]
            ),
        ],
        widths=[None, 1],
    )
    return i_vals, mean_energy, probs


@app.cell(hide_code=True)
def _(i_vals, mean_energy, mo, plt, probs):
    # Probability distribution
    plt.figure()
    plt.bar(i_vals, probs)
    plt.axvline(mean_energy, linestyle='--', label=f"Mean = {mean_energy:.2f}")
    plt.title("Probability Distribution of Energy States")
    plt.xlabel("Energy in Subsystem 1 (i)")
    plt.ylabel("Probability")
    plt.legend()
    mo.center(plt.gca())
    return


@app.cell(hide_code=True)
def _(i_vals, mean_energy, mo, plt, probs):
    # Cumulative Distribution
    cdf = probs.cumsum()
    plt.figure()
    plt.step(i_vals, cdf, where='mid')
    plt.axvline(mean_energy, linestyle='--', label=f"Mean = {mean_energy:.2f}")
    plt.title("Cumulative Distribution of Energy States")
    plt.xlabel("Energy in Subsystem 1 (i)")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    mo.center(plt.gca())
    return


if __name__ == "__main__":
    app.run()
