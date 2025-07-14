import marimo

__generated_with = "0.14.10"
app = marimo.App()


@app.cell
def _():
    import matplotlib.pyplot as plt
    import math
    import numpy as np
    from random import random
    from copy import deepcopy
    from prettytable import PrettyTable

    # Energy of harmonic oscillator
    def energy(q):
        return q**2/2


    # Probability of the configuration
    def prob(q):
        return math.e ** (-q**2/2)


    # Compute the trial position q_old ± DELTA
    def trial(q, delta):
        return q + random() * 2 * delta - delta
    return PrettyTable, deepcopy, math, np, plt, prob, random, trial


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The function that runs the simulation
        Trajectory away from the equilibrium position decay linearly to the equilibrio
        """
    )
    return


@app.cell
def _(prob, random, trial):
    def run(q=4, steps=1000, delta=0.1):
        p = prob(q)
        positions = []
        accepted = 0
        q_squared = 0
        for _ in range(steps):
            qtry = trial(q, delta)
            ptry = prob(qtry)
            alpha = ptry / p
            if alpha > random():
                q = qtry
                p = ptry
                accepted = accepted + 1
            positions.append(q)
            q_squared = q_squared + q ** 2
        return (positions, round(accepted / steps, 3), round(q_squared / steps, 3))
    return (run,)


@app.cell
def _(PrettyTable, math, np, plt, prob, run):
    steps = [1, 0.1, 0.01]
    lengths = [1000000, 100000, 10000, 1000]
    table = PrettyTable()
    table.align = 'l'
    table.field_names = ['# Steps', 'Delta', 'Acceptance', '<q^2>', 'Error on <q^2> (%)']
    for step in steps:
        positions_steps = []
        acceptances = []
        q_squareds = []
        for i in range(0, len(lengths)):
            sim = run(2, lengths[i], step)
            positions_steps.append(sim[0])
            acceptances.append(sim[1])
            q_squareds.append(sim[2])
        fig, axes = plt.subplots(1, len(lengths), figsize=(18, 5), sharey=True)
        datasets = [(positions_steps[0], f'1E6 - {step}'), (positions_steps[1], f'1E5 - {step}'), (positions_steps[2], f'1E4 - {step}'), (positions_steps[3], f'1E3 - {step}')]
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'darkorchid']
        for ax, (_data, title), color in zip(axes, datasets, colors):
            count, bins, _ = ax.hist(_data, bins=30, density=True, color=color, alpha=0.6, edgecolor='black')
            _points = np.linspace(-4, 4, 500)
            ax.plot(_points, prob(_points) / math.sqrt(2 * math.pi), 'k-', linewidth=2)
            ax.set_title(title)
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
        plt.tight_layout()
        plt.show()
        for i in range(0, len(lengths)):
            table.add_row([lengths[i], step, acceptances[i], q_squareds[i], round(abs(q_squareds[i] - 1) * 100, 3)], divider=i == len(lengths) - 1)
    print(table)
    return


@app.cell
def _(np, traj):
    # Block analysis
    block_sizes = [1,2,5,10,20,50,100,200,500,1000,2000,5000]

    for bs in block_sizes:
        number_of_blocks = len(traj) / bs
        if len(traj) * 2 >= bs:
            continue
        block_averages = np.average(traj.reshape((number_of_blocks, bs))**2, axis=1)
        np.std(block_averages)/(np.sqrt(number_of_blocks-1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        3D Coupled atoms
        The wrong gaussian distribution doesnt account for state density. Number of states grows with the square of the distance in 3d
        """
    )
    return


@app.cell
def _(deepcopy, math, np, plt, random):
    class Vec3(object):

        def __init__(self, x, y, z):
            self.vec = [x, y, z]

        def __repr__(self):
            return '{self.__class__.__name__}(x={self.vec[0]}, y={self.vec[1]}, z={self.vec[2]})'.format(self=self)

        def __add__(self, other):
            x1, y1, z1 = self.vec
            x2, y2, z2 = other.vec
            return Vec3(x1 + x2, y1 + y2, z1 + z2)

        def __sub__(self, other):
            x1, y1, z1 = self.vec
            x2, y2, z2 = other.vec
            return Vec3(x1 - x2, y1 - y2, z1 - z2)

        def __getitem__(self, key):
            return self.vec[key]

        def __setitem__(self, key, value):
            self.vec[key] = value

        def norm(self):
            return math.sqrt(sum((x ** 2 for x in self.vec)))

        def dot(self, other):
            x1, y1, z1 = self.vec
            x2, y2, z2 = other.vec
            return x1 * x2 + y1 * y2 + z1 * z2

    def prob_1(q1, q2):
        return math.exp(-0.5 * ((q1 - q2).norm() - 4) ** 2)

    def run_1(qinit=[Vec3(-2, 0, 0), Vec3(2, 0, 0)], steps=10000, delta=1):
        distances = []
        p = prob_1(qinit[0], qinit[1])
        accepted = 0
        for i in range(steps):
            qtry = deepcopy(qinit)
            particle = int(random() * 2)
            direction = int(random() * 3)
            qtry[particle][direction] = qtry[particle][direction] + (random() * 2 * delta - delta)
            ptry = prob_1(qtry[0], qtry[1])
            alpha = ptry / p
            if alpha > 1 or alpha > random():
                qinit = qtry
                p = ptry
                accepted = accepted + 1
            distances.append((qinit[0] - qinit[1]).norm())
        print('ACCEPTANCE: ', round(accepted / steps, 3))
        return distances
    _data = run_1()
    print('AVG: ', round(sum(_data) / len(_data), 3))
    plt.plot(np.arange(0, len(_data)), _data)
    plt.show()
    plt.hist(_data, bins=30, density=True, color='coral', alpha=0.6, edgecolor='black')
    _points = np.linspace(1, 8, 500)
    gaussians = []
    for point in _points:
        gaussians.append(point ** 2 * math.exp(-(point - 4) ** 2 / 2) / 106 * math.sqrt(2 * math.pi))
    plt.plot(_points, gaussians, 'k-', linewidth=2)
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
