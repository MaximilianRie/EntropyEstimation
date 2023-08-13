from jax import vmap
import jax.random as random
import jax.numpy as jnp
import flax
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
from models import RNNGATEntropyEstimator
from load_data import load_results

# hyperparameters for plot layout
mpl.rcParams["figure.autolayout"] = True
mpl.rcParams["font.size"] = 28
mpl.rcParams["xtick.labelsize"] = 28
mpl.rcParams["ytick.labelsize"] = 28
mpl.rcParams["axes.labelsize"] = 28
mpl.rcParams["lines.linewidth"] = 4
mpl.rcParams["lines.markersize"] = 15

# some custom colors
red = "#db5f57"
yellow = "#dac357"
light_green = "#91db58"
dark_blue = "#5770db"
magenta = "#db57b2"
turquoise = "#57d3db"
violett = "#a256db"
green = "#58db80"

# metadata of project directory (change this to your own DIR) and 
# name of file
DIR = "/p/project/neuralqmb/maximilian/EntropyEstimation/"
NAME = "Ising-10-spins--TIME--100-states-50000-samples"
INFO = "-MCMC-batchsize1000uni+lin-Renyi-from0.0-to5.0"
VERSION = ""

# load data, labels and theta (e.g. time, magnetization) of train.
# and val. set.
theta, data, labels, val_theta, \
            val_data, val_labels, doc = load_results(DIR, NAME, INFO, VERSION,
                                                data_indices=jnp.arange(50),
                                                val_data_indices=jnp.array([22]),
                                                label_func=None)

# usually bad practice, but convenient here
locals().update(doc)

# initialize model
model = RNNGATEntropyEstimator(FEATURES_RNN, FEATURES_GAT, FEATURES_RHO,
                               num_samples=NUM_BATCH_SAMPLES,
                               num_heads=NUM_HEADS,
                               avg_func=jnp.mean)

vmodel = vmap(model.apply, in_axes=(None, 0))

# import network parameters (model initialization was necessary for this)
with open(DIR + "results/" + NAME + INFO + VERSION + "-best_params", "rb") as f:
    params = model.init(random.PRNGKey(0), data[0, 0])
    params = flax.serialization.from_bytes(params, f.read())


# evalaute network on (val) data
pe_mean, pe_std = vmodel(params, data[:, 22, :]).T
vpe_mean, vpe_std = vmodel(params, val_data).T

# plot results
fig = plt.figure(figsize=(4*(jnp.sqrt(5) + 1), 8))
ax = fig.gca()
ax.grid(visible=True)

ax.plot(theta, labels, color=red, label="labels")
ax.scatter(theta, pe_mean, color=light_green,
           label="training set", marker="x")
ax.errorbar(val_theta, vpe_mean, yerr=vpe_std, color=dark_blue,
           label="validation set", fmt=".")
plt.xlabel(r"time $ht$")
plt.ylabel(r"Rényi entropy $S^{(2)}$")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False)
plt.savefig(DIR + "results/demo.png", dpi=300)
