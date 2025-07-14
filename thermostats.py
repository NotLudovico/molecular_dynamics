import marimo

__generated_with = "0.14.10"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Assignment

        Perform a simulation of a Lennard Jones crystal with 864 particles. Temperature should be controlled using the Langevin thermostat at temperature T = 2. Try with different values of the friction γ (e.g. 0.01, 0.1, 1.0,10.0) and look at the difference in the behavior of the potential energy
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
