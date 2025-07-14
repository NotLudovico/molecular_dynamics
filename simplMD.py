import marimo

__generated_with = "0.14.10"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # STEP 0 - SETTING UP THE ENVIRONMENT

        Run the next few cells to import some needed modules and set up your working directory



        """
    )
    return


app._unparsable_cell(
    r"""
    # import generic modules
    import sys
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    # this is a module that can be used to visualize trajectories
    try:
      import py3Dmol
    except:
      # only install it if it's not already present
      !pip install py3Dmol
    """,
    name="_"
)


@app.cell
def _():
    # use local directory, will be overridden in the next section if you are using colab
    path="."
    return (path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Now we will connect google colab to our own google drive

        The goal of this part is to make sure you can save files in a place where you can find them if you open colab later. To do so you need to connect this sheet to your google drive, and you will need some space there.
        **This is only necessary if you want to work within colab. If you work locally on your computer (e.g., opening this notebook with VSCode or Jupyter lab) it is not needed, all the files that you save will be on your hard disk**

        Run the following cell and log in with your google account.
        """
    )
    return


@app.cell
def _():
    from google.colab import drive
    drive.mount('/content/drive')
    return


app._unparsable_cell(
    r"""
    # go to your drive
    os.chdir('/content/drive/MyDrive')

    # create working directory
    # -p only creates it if it does not exist already
    os.makedirs('-p md_exercises', exist_ok=True)
    # go to directory
    os.chdir('md_exercises')

    # check the path of the folder. should be something like
    # /content/drive/MyDrive/md_exercises
    # magic command not supported in marimo; please file an issue to add support
    # %pwd 

    # save path to a local variable, will be useful later
    path = %pwd
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        From now on, you can treat your drive as a local directory and
        save and access to data in it, here is a simple example. You can also browse your data by clicking on the folder symobl on your right and goind "manually" to your folder.
        """
    )
    return


@app.cell
def _(np, path):
    # create a random array with numpy
    random_array = np.random.random((10,2))

    # save the array in your folder
    np.savetxt(path+"/random_array.txt",random_array)

    # access to the array
    print(np.loadtxt(path+"/random_array.txt"))
    return


app._unparsable_cell(
    r"""
    !ls /content/drive/MyDrive/md_exercises
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # We download here the *simplemd* code for Molecular Dynamics

        After running this cell a folder containing the python code for running simple md will be created in your working directory. We will use as a starting point the pure python version that was prepared for a cecam summer school in 2024. This is why we will download branch cecam2024
        """
    )
    return


app._unparsable_cell(
    r"""
    # download simplemd
    !test -d simplemd || git clone --branch cecam2024 https://github.com/GiovanniBussi/simplemd.git
    #sys.path.append(os.getcwd() + \"/simplemd/python\")
    import simplemd
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # STEP 1 - LEARN TO RUN *simplemd*
        *simplemd.SimpleMD is a class, to be initialized it requires a dictionary with the input parameters. Use Lennard-Jones units for the parameters.*


        1. generate some initial positions and a simulation box using the method generate_lattice(n), it generates $4n^3$ particles and a simulation box. Print the shape of the initial positions array and the simulation box to check how they are made.

        *hint 1. generate positions and simulation box with the method generate_lattice(n) of the simplemd module (remember the sintax is module.method(parameters)). It outputs the position of $4\times n \times n \times n$ particles and the coordinates of the simulation box, a good number is $\mathcal{O}(10^2)$ total particles.*


        2.   create a dictionary specifying the following keys:

        *   "temperature"
        *   "tstep"


        *   "forcecutoff" (radius within which I compute the forces)


        *   "listcutoff" (radius within which I compute the lists)
        *   "nstep"


        *   "nconfig" (tells how often I save the positions)
        *   "nstat" (tells how often I save the statistics of the simulation)



        *   "cell" (simulation box)
        *   "positions" (starting configuration)


        *hint 2. start experimenting with a small nstep (order of 10^3), listcutoff = forcecutoff + 0.5, listcutoff < cell size*


        3. Create a runner: assign to a variable the initialized class SimpleMD and run your first Molecular Dynamics simulation using runner.run()

        *hint 3. the sintax is runner_variable = module.class(**dictionary_name), run with runner_variable.run(). **dictionary_name is used to pass the full dictionary to the class.*

        4. after running you will be able to access the *statistics* and the *trajectory* using the attributes statistics and trajectory associated to your runner variable. Use the statistics and make a plot of the timeseries of: instantaneous temperature (add also a horizontal line with the chosen temperature), potential energy, total energy. What do you expect to see when comparing the fixed temperature and the instantaneous temperature?

        *hint 4. the syntax to access the attributes is runner_variable.attribute. Remember: statistics is a (6, n_steps_saved) matrix and contains in the following order: istep, time, instantaneous temperature, potential energy, total energy, total energy + Δenergy integration.*
        """
    )
    return


@app.cell
def _(simplemd):
    # solution 1.
    cell, positions = simplemd.generate_lattice(3)
    print(positions.shape)
    print(cell)
    return cell, positions


@app.cell
def _(cell, positions):
    # solution 2.
    # reasonable input options
    # note: initial coordinates are still missing
    keys={
     'temperature': 0.722,
     'tstep': 0.005,
     'forcecutoff': 2.5,
     'listcutoff': 3.0,
     'nstep': 2000,
     'nconfig': 10,
     'nstat': 10
     }
    # generate a crystal lattice and pass it:
    keys["cell"] = cell
    keys["positions"] = positions
    return (keys,)


@app.cell
def _(keys, simplemd):
    # solution 3.
    # create runner
    smd=simplemd.SimpleMD(**keys)
    # run
    smd.run()
    return (smd,)


@app.cell
def _(keys, np, plt, smd):
    # solution 4.
    # plotting instantaneous temperature
    fig, ax = plt.subplots()
    ax.plot(np.array(smd.statistics).T[2])
    ax.axhline(y=keys["temperature"], color='r', linestyle='--')
    ax.set_xlabel('step')
    ax.set_ylabel('temperature')
    plt.show()

    # plotting potential energy
    fig, ax = plt.subplots()
    ax.plot(np.array(smd.statistics).T[3], label="potential energy")
    ax.plot(np.array(smd.statistics).T[4], label="total energy")
    ax.legend()
    ax.set_xlabel('step')
    ax.set_ylabel('energy')
    plt.show()

    # Temperature <=> Kinetic energy, drops down because particles are placed in minima of potential
    # Energy stabilizes around "half" of total energy due to equipartition theorem
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # STEP 2 - VISUALIZATION OF THE TRAJECTORIES

        For visualizing the trajectories we will use the pyhton module "py3Dmol" that we have loaded at the beginning of this exercise, and the method "write_trajectory" of the simplemd module.

        1. generate a starting configuration with 108 particles (n=3) and and a simulation box. Initialize the SimpleMD class with the initial configuration, the cell and  'temperature': 0.722,
         'tstep': 0.005,
         'forcecutoff': 2.5,
         'listcutoff': 3.0,
         'nstep': 10000,
         'nconfig': 10,
         'nstat': 10,
         'cell': your cell,
         'positions': your starting configuration.

          Create a runner and launch a simulation.

        2. Generate a text file with the trajectory using the method write_trajectory of the simplemd module and save it to your working directory as a .xyz file.

        *hint 1. the syntax is simplemd.write_trajectory(path+"name_to_save.xyz", runner_variable.trajectory)*

        3. visualize the trajectory using py3Dmol with the following steps:


        *   open the saved trajectory and process it as a text file

        *hint 2. the syntax is the following:*

        with open(path+"/name_saved_trajectory.xyz") as f:

        traj_xyz = f.read()

        *   visualize your text file trajectory using py3Dmol, here is a detailed description of its functioning https://william-dawson.github.io/using-py3dmol.html, https://pypi.org/project/py3Dmol/

        *hint 3. you can use the following syntax:*



        *   view = py3Dmol.view(width=400, height=300)

        this creates the object that contains the visualization window (in this case a box 400x300 pixels)
        *   view.addModelsAsFrames(traj_xyz, "xyz")

        this adds the frames contained in the trajectory to the visualization box


        *   view.setStyle({"sphere": {"radius":0.4}})

        this will set the particles to be visualized as spheres
        *   view.animate({'loop': "forward"})


        this will create a looping video out of your trajectory




        *   view.zoomTo()
        *   view.show()

        these last two lines will display the movie.



        4.   Check the movie that you have created, do you notice something unusual?
        5.   Redo the previous steps with a initial temperature = 4, how does the movie change?






        """
    )
    return


@app.cell
def _(simplemd):
    keys_1 = {'temperature': 1.3, 'friction': 1.0, 'tstep': 0.005, 'forcecutoff': 2.5, 'listcutoff': 3.0, 'nstep': 10000, 'nconfig': 10, 'nstat': 10}
    cell_1, positions_1 = simplemd.generate_lattice(3)
    keys_1['cell'] = cell_1
    keys_1['positions'] = positions_1
    smd_1 = simplemd.SimpleMD(**keys_1)
    smd_1.run()
    return (smd_1,)


@app.cell
def _(path, simplemd, smd_1):
    simplemd.write_trajectory(path + '/trajectory.xyz', smd_1.trajectory)
    return


@app.cell
def _(path, py3Dmol):
    #needs to be processed as text
    with open(path+"/trajectory.xyz") as f:
        traj_xyz = f.read()

    view = py3Dmol.view(width=400, height=300)
    view.addModelsAsFrames(traj_xyz, "xyz")
    view.setStyle({"sphere": {"radius":0.4}})
    view.animate({'loop': "forward"})
    view.zoomTo()
    view.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # ASSIGNMENT

        1. generate a cell and a starting configuration with 108 atoms. Re-initialize the SimpleMD class with the following parameters:

         'temperature': 0.722,
         'tstep': 0.005,
         'forcecutoff': 2.5,
         'listcutoff': 3.0,
         'nstep': 2000,
         'nconfig': 10,
         'nstat': 10
         'cell': your simulation box
         'positions': your starting configuration

        now check how the energy conservation changes by running trajectories using different timesteps and find the maximum timestep allowed for this system, at each different run save the statistics to your working directory using np.savetxt(...) so you can access them in the future.

        *hint 1. you can check energy conservation visually simply by plotting the total energy in multiple lines as a function of time and checking if there is a drift, you can also plot the fluctuations of the total energy and potential energy as a function of the timestep. You can use timesteps such as 0.0001, 0.001, 0.01 or larger. You may want to rescale the number of steps according to the timestep to have equally long simulations in time.*

        2. Now fix the tstep to 0.001 and repeat the analysis with different values of the temperature (e.g. T=0.01, T=0.1, T=1, T=2), plot in particular the total and potential energy, what do you see?

        3. Now fix the temperature = 0.722, the tstep = 0.005 and investigate the system size effect, by running the same analysis for 32, 108, 500 particles. What do you see? Does the maximum timestep allowed change, try for example for 500 particles,
        (simplemd.generate_lattice(4)).


        """
    )
    return


@app.cell
def _(np, simplemd):
    total_time = 10
    dts = [0.0001, 0.001, 0.005]
    temps = [0.722, 0.1, 1, 2]
    for temp in temps:
        for dt in dts:
            keys_2 = {'temperature': temp, 'tstep': dt, 'forcecutoff': 2.5, 'listcutoff': 3.0, 'nstep': int(total_time / dt), 'nconfig': 10, 'nstat': 10}
            cell_2, positions_2 = simplemd.generate_lattice(3)
            keys_2['cell'] = cell_2
            keys_2['positions'] = positions_2
            smd_2 = simplemd.SimpleMD(**keys_2)
            smd_2.run()
            stat_name = f'./stat/{dt}-{temp}-stat.txt'
            np.savetxt(stat_name, smd_2.statistics)
    return dts, temps


@app.cell
def _(dts, np, plt, temps):
    for temp_1 in temps:
        fig_1, ax_1 = plt.subplots()
        for dt_1 in dts:
            stat_name_1 = f'./stat/{dt_1}-{temp_1}-stat.txt'
            vals = np.genfromtxt(stat_name_1)
            ax_1.plot(np.arange(0, int(1 / dt_1), 1) * (dt_1 / 0.001), np.array(vals).T[4], label=f'{dt_1}-{temp_1}')
        ax_1.legend()
    return


@app.cell
def _(np, plt, temps):
    for temp_2 in temps:
        fig_2, ax_2 = plt.subplots()
        stat_name_2 = f'./stat/0.001-{temp_2}-stat.txt'
        vals_1 = np.genfromtxt(stat_name_2)
        ax_2.plot(np.array(vals_1).T[3], label='potential energy')
        ax_2.plot(np.array(vals_1).T[4], label='total energy')
        title = f'Temp: {temp_2}'
        ax_2.set_title(title)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
         Now fix the temperature = 0.722, the tstep = 0.005 and investigate the system size effect, by running the same analysis for 32, 108, 500 particles. What do you see? Does the maximum timestep allowed change, try for example for 500 particles,(simplemd.generate_lattice(4)).
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
