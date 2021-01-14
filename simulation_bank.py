import numpy as np
import matplotlib.pyplot as plt
import time

from simulation import InteractionModel
from data_collection import compute_average_performance

from mesa.batchrunner import BatchRunner



def sim_one_run(N=15, omega=0.5, w=0.5, alpha=0.5, beta=0.1, iters=200, max_steps=300 ):

    runs =[]
    for iter_ in range(iters):
        sim = InteractionModel(N=N, omega=omega, w=w,alpha=alpha, beta=beta)
        for step_ in range(max_steps):
            sim.step()
        runs.append( sim.datacollector.get_model_vars_dataframe())

    return runs






def sim_w_beta(w, beta,alpha =0.5, omega =0.5, N=15, iters=200, max_steps=300 ):
    fixed_params = {
        "alpha": alpha,
        "omega": omega,
        "N": N,
    }
    variable_params = {
        "w": w,
        "beta": beta,
    }

    batch_run = BatchRunner(
        InteractionModel,
        variable_params,
        fixed_params,
        iterations=iters,
        max_steps= max_steps,
        model_reporters={"AP": compute_average_performance}
    )

    run_name = f"base_model_{batch_run.iterations}_rep_{int(time.time())}_beta_w"

    batch_run.run_all()
    run_data = batch_run.get_model_vars_dataframe()
    grouped = run_data.groupby("w", as_index=False)

    data = grouped.aggregate(np.average)
    plt.scatter(data.w, data.AP)
    plt.savefig("./data/" + run_name + "_scatter.png")
    run_data.to_csv("./data/" + run_name)

    df = data.loc[:, ['w', 'AP']]
    df.rolling(10).mean().plot.scatter(x='w', y='AP');
    plt.savefig("./data/" + run_name + "_smoothed.png")

    return run_data, run_name
