from functools import partial
import jax
from jax import jit, vmap
import jax.random as random
import jax.numpy as jnp
import flax
import optax
import h5py
from models import RNNGATEntropyEstimator
from load_data import load_pure_data
import sys


################################
#                              #
#       global variables       #
#                              #
################################


# metadata and location of datafiles
# DIR should be project directory
DIR = "/p/project/neuralqmb/maximilian/EntropyEstimation/"
NAME = "Ising-10-spins--TIME--100-states-50000-samples"
INFO = "-MCMC-batchsize1000uni+lin-Renyi-from0.0-to5.0"
VERSION = ""

# RNG keys for training
try:
    enum = sys.argv[1]
    KEY = random.PRNGKey(int(enum))
    print("detected rng key with seed:", enum)
except IndexError:
    KEY = random.PRNGKey(2207)
key_init, key_train, key_bootstrap = random.split(KEY, 3)
FILES = [
    "data/Ising-10-spins--TIME--100-states-50000-samples-MCMC-batchsize1000uni+lin-Renyi-from0.0-to5.0"
]


# all data gets padded to MAX_SYSTEM_SIZE in order to be stored in the same array
# this is relevant if the network is trained on different system sizes
MAX_SYSTEM_SIZE = 10


################################
#                              #
#        importing data        #
#                              #
################################

# calculate the partial trace of density matrix
def partial_trace(state, subsys):
    dim = int(jnp.sqrt(len(state)))
    # rho has to be calculated if state is a vec not a mat
    rho = jnp.outer(state.conj(), state)
    M = rho.reshape(dim, dim, dim, dim)
    if subsys == "A":
        ptrace = jnp.einsum("ijik->jk", M)
    elif subsys == "B":
        ptrace = jnp.einsum("jiki->jk", M)
    return ptrace

# calculate second order Renyi entropy of density matrix
def RenyiS(state):
    return -jnp.log(jnp.real(jnp.trace(state@state)))

# calculate mutual information of density matrix between subsystems
def label_func(state):
    return RenyiS(partial_trace(state, "B")) \
            + RenyiS(partial_trace(state, "A")) - RenyiS(state)
lf = vmap(label_func, in_axes=0)

# load data, labels and theta (e.g. time, magnetization) of train.
# and val. set.
theta, data, labels, val_theta, val_data, \
        val_labels = load_pure_data(DIR, FILES, MAX_SYSTEM_SIZE,
                                    data_indices=jnp.arange(50),
                                    val_data_indices=jnp.array([22]),
                                    label_func=None)

# useful variables for later
NUM_STATES, NUM_BATCHES, NUM_BATCH_SAMPLES, N, NUM_POVM = data.shape
NUM_TRAIN = NUM_STATES * NUM_BATCHES
NUM_SAMPLES = NUM_BATCH_SAMPLES * NUM_BATCHES

print("shape of half chain dataset:", data.shape)
print("shape of half chain validation dataset:", val_data.shape)


################################
#                              #
#  model parameters and setup  #
#                              #
################################


# features of inner networks
FEATURES_GAT = [10, 10]
FEATURES_RNN = [20, 20, 20]
# features of outer network
FEATURES_RHO = [4, 2]
# number of heads for GAT model
NUM_HEADS = 1
# number of epochs to train
NUM_EPOCHS = 10
# learning rate
LEARNING_RATE = 0.0005
# size of minibatches to pass into the network at once
MINIBATCH_SIZE = 5

# usefull dict to store global variables later when saving results
doc = {"NUM_STATES": NUM_STATES, "NUM_BATCHES": NUM_BATCHES,
       "NUM_BATCH_SAMPLES": NUM_BATCH_SAMPLES, "NUM_POVM": NUM_POVM,
       "N": N, "FEATURES_GAT": FEATURES_GAT, "FEATURES_RHO": FEATURES_RHO,
       "FEATURES_RNN": FEATURES_RNN, "NUM_HEADS": NUM_HEADS,
       "NUM_EPOCHS": NUM_EPOCHS, "LEARNING_RATE": LEARNING_RATE,
       "MINIBATCH_SIZE": MINIBATCH_SIZE, "MAX_SYSTEM_SIZE": MAX_SYSTEM_SIZE}


@partial(jit, static_argnums=3)
def loss(params, data, labels, vmodel):
    """loss function"""
    mu, sig = vmodel(params, data).T
    l = jnp.mean(((mu - labels)**2)/(2*sig**2) + jnp.log(sig))
    return l


def get_batches(key, batch_size):
    """shuffles data and labels and seperates them into minibatches of given size"""
    key1, key2 = random.split(key)
    _data = random.permutation(key1, data.reshape(NUM_STATES,
                                                  NUM_SAMPLES,
                                                  N, NUM_POVM), axis=1)
    _data = random.permutation(key2, _data.reshape(NUM_TRAIN,
                                                   NUM_BATCH_SAMPLES,
                                                   N, NUM_POVM),
                               axis=0)
    _labels = random.permutation(key2, labels.repeat(NUM_BATCHES))
    _data = _data.reshape(NUM_TRAIN//batch_size, batch_size,
                          NUM_BATCH_SAMPLES,
                          N, NUM_POVM)
    _labels = _labels.reshape((NUM_TRAIN//batch_size, batch_size))
    return zip(_data, _labels)


loss_grad_fn = jax.value_and_grad(loss)
@partial(jit, static_argnums=(0, 5))
def update_step(optimizer, opt_state, params, data, labels, vmodel):
    """return updated parameters after gradient descend (or other optimization)"""
    train_loss, grads = loss_grad_fn(params, data, labels, vmodel)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return opt_state, params, train_loss

# initialize model and parameters
model = RNNGATEntropyEstimator(FEATURES_RNN, FEATURES_GAT, FEATURES_RHO,
                               num_samples=NUM_BATCH_SAMPLES,
                               num_heads=NUM_HEADS)
params = model.init(key_init, data[0, 0])
# vectorize model
vmodel = vmap(model.apply, in_axes=(None, 0))

# initialize optimizer
optimizer = optax.adam(learning_rate=LEARNING_RATE)
opt_state = optimizer.init(params)


################################
#                              #
#           training           #
#                              #
################################


train_losses = []
val_losses = []

for epoch in range(1, NUM_EPOCHS+1):
    key_train, _ = random.split(key_train)
    for tdb, tlb in get_batches(key_train, MINIBATCH_SIZE):
        opt_state, params, train_loss = update_step(optimizer, opt_state,
                                                    params, tdb, tlb,
                                                    vmodel)
    val_loss = loss(params, val_data, val_labels, vmodel)
    train_loss = loss(params, data.reshape(NUM_TRAIN, NUM_BATCH_SAMPLES,
                                           N, NUM_POVM),
                                           labels.repeat(NUM_BATCHES),
                                           vmodel)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    if val_loss <= min(val_losses):
        best_params = params
    if epoch % 10 == 0:
        print("epoch:", epoch, "\n",
              "train loss:", train_loss, "\n",
              "val loss:  ", val_loss, "\n")

print("training finished")
train_losses = jnp.array(train_losses, dtype=jnp.float32)
val_losses = jnp.array(val_losses, dtype=jnp.float32)
print("converted losses to jax arrays")


################################
#                              #
# evaluating and saving result #
#                              #
################################

try:
    VERSION += "-e_num" + enum
except NameError:
    print("running in 'single enseble' mode")


SAVEFILE = DIR + "results/" + NAME + INFO + \
        VERSION + "-RNNresults.hdf5"

with h5py.File(SAVEFILE, "w") as f:
    f.create_dataset("train_losses", data=train_losses)
    print("saved train losses")
    f.create_dataset("val_losses", data=val_losses)
    print("saved train val_losses")
    f.create_dataset("FILES", data=[f.encode("utf8") for f in FILES])
    for k in doc.keys():
        f.create_dataset(k, data=doc[k])
    print("saved DOC")

print("saved results in:", SAVEFILE)


with open(DIR + "results/" + NAME + INFO + \
        VERSION + "-params", "wb") as f:
    f.write(flax.serialization.to_bytes(params))

print("saved final parameters")

with open(DIR + "results/" + NAME + INFO + \
        VERSION + "-best_params", "wb") as f:
    f.write(flax.serialization.to_bytes(best_params))

print("saved best parameters")

print("finished...")
