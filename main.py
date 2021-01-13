"""
Main file just for completeness sake, use the run.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from mesa.batchrunner import BatchRunner

from simulation import InteractionModel
from data_collection import compute_average_performance

fixed_params = {
    "beta": 0.0,
    "alpha": 0.5,
    "omega": 0.5,
    "N": 15,
}
variable_params = {
    "w": np.arange(0.01, 1.01, 0.01)
}

batch_run = BatchRunner(
    InteractionModel,
    variable_params,
    fixed_params,
    iterations=400,
    max_steps=300,
    model_reporters={"AP": compute_average_performance}
)


run_name = fr"run_iter_{batch_run.iterations}_rep_{datetime.now()}_beta_{fixed_params['beta']}"

batch_run.run_all()
run_data = batch_run.get_model_vars_dataframe()
grouped = run_data.groupby("w", as_index=False)
data = grouped.aggregate(np.average)
plt.scatter(data.w, data.AP)
plt.show()
plt.savefig("./data/" + run_name+".png")
run_data.to_csv("./data/"+run_name)

df = data.loc[:,['w', 'AP']]
df.rolling(10).mean().plot.scatter(x='w', y='AP'); plt.show()
