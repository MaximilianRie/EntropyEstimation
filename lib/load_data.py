import jax.random as random
import jax.numpy as jnp
import flax
import h5py
from models import RNNGATEntropyEstimator

def load_pure_data(DIR, FILES, MAX_SYSTEM_SIZE, data_indices=jnp.arange(10),
                   val_data_indices=jnp.array([22]), label_func=None,
                   with_states=False):
    # iterate through datafiles and concatenate data
    for (i, filename) in enumerate(FILES):
        with h5py.File(DIR + filename + ".hdf5", "r") as f:
            # select data up to 'data_indices', if not all data is used.
            _data = jnp.array(f["data"][:])[:, data_indices].squeeze()
            _theta = f["theta"][:]
            _states = f["states"][:]
            # if label_func is given, determined labels from states. Otherwise load labels.
            if label_func:
                _labels = label_func(_states)
            else:
                _labels = f["labels"][:]

            # select val. data according to 'val_data_indices'.
            _val_data = jnp.array(f["val_data"][:])[:, val_data_indices].squeeze()
            _val_theta = f["val_theta"][:]
            _val_states = f["val_states"][:]
            if label_func:
                _val_labels = label_func(_val_states)
            else:
                _val_labels = f["val_labels"][:]


            # system size
            N = _data.shape[-2]
            # info entry of system size
            Narr = N * jnp.ones(shape=(*_data.shape[0:-2], 1,
                                       _data.shape[-1]))
            # padding to MAX_SYSTEM_SIZE
            padding = jnp.zeros(shape=(*_data.shape[0:-2],
                                       MAX_SYSTEM_SIZE-N,
                                       _data.shape[-1]))
            # add padding and system size info
            _data = jnp.concatenate((_data, padding, Narr), axis=-2)
            Narr = N * jnp.ones(shape=(*_val_data.shape[0:-2], 1,
                                       _val_data.shape[-1]))
            padding = jnp.zeros(shape=(*_val_data.shape[0:-2],
                                       MAX_SYSTEM_SIZE-N,
                                       _val_data.shape[-1]))
            _val_data = jnp.concatenate((_val_data, padding, Narr), axis=-2)

        if i == 0:
            data = _data
            labels = _labels
            theta = _theta
            val_data = _val_data
            val_labels = _val_labels
            val_theta = _val_theta

        else:
            data = jnp.concatenate((data, _data), axis=0)
            labels = jnp.concatenate((labels, _labels), axis=0)
            theta = jnp.concatenate((theta, _theta), axis=0)
            val_data = jnp.concatenate((val_data, _val_data), axis=0)
            val_labels = jnp.concatenate((val_labels, _val_labels), axis=0)
            val_theta = jnp.concatenate((val_theta, _val_theta), axis=0)

    if not with_states:
        return theta, data, labels, val_theta, val_data, val_labels
    return theta, data, labels, val_theta, val_data, val_labels, _states, _val_states


def load_results(DIR, NAME, INFO, VERSION, data_indices, val_data_indices,
                 label_func=None):
    with h5py.File(DIR + "results/" + NAME + INFO + \
            VERSION + "-RNNresults.hdf5", "r") as f:
        NUM_STATES = f["NUM_STATES"][()]
        NUM_BATCHES = f["NUM_BATCHES"][()]
        NUM_BATCH_SAMPLES = f["NUM_BATCH_SAMPLES"][()]
        N = f["N"][()]
        NUM_POVM = f["NUM_POVM"][()]
        train_losses = f["train_losses"][:]
        val_losses = f["val_losses"][:]
        FEATURES_RNN = f["FEATURES_RNN"][()]
        FEATURES_GAT = f["FEATURES_GAT"][()]
        FEATURES_RHO = f["FEATURES_RHO"][()]
        FILES = f["FILES"][()].astype("str")
        MAX_SYSTEM_SIZE = f["MAX_SYSTEM_SIZE"][()]
        WITH_N = f["WITH_N"][()]
        NUM_HEADS = f["NUM_HEADS"][()]
        NUM_EPOCHS = f["NUM_EPOCHS"][()]
        LEARNING_RATE = f["LEARNING_RATE"][()]
        MINIBATCH_SIZE = f["MINIBATCH_SIZE"][()]

    doc = {"NUM_STATES": NUM_STATES, "NUM_BATCHES": NUM_BATCHES,
           "NUM_BATCH_SAMPLES": NUM_BATCH_SAMPLES, "NUM_POVM": NUM_POVM,
           "N": N, "FEATURES_GAT": FEATURES_GAT, "FEATURES_RHO": FEATURES_RHO,
           "FEATURES_RNN": FEATURES_RNN, "NUM_HEADS": NUM_HEADS,
           "NUM_EPOCHS": NUM_EPOCHS, "LEARNING_RATE": LEARNING_RATE,
           "MINIBATCH_SIZE": MINIBATCH_SIZE, "MAX_SYSTEM_SIZE": MAX_SYSTEM_SIZE,
           "WITH_N": WITH_N, "train_losses": train_losses,
           "val_losses": val_losses, "FILES": FILES}

    theta, data, labels, val_theta, \
            val_data, val_labels = load_pure_data(DIR, FILES, MAX_SYSTEM_SIZE,
                                                  data_indices,
                                                  val_data_indices,
                                                  label_func=label_func)
    return theta, data, labels, val_theta, val_data, val_labels, doc

def init_ensemble_from_results(DIR, NAME, INFO, VERSION, KEY_SEEDS,
        data_indices, val_data_indices, label_func=None):
    ens_params = []
    ens_best_params = []
    ens_losses = []
    ens_val_losses = []
    for KEY_SEED in KEY_SEEDS:
        _VERSION = VERSION + "-e_num" + str(KEY_SEED)
        theta, data, labels, val_theta, \
        val_data, val_labels, doc = load_results(DIR, NAME, INFO, _VERSION,
                                            data_indices=data_indices,
                                            val_data_indices=val_data_indices,
                                            label_func=label_func)

        ens_losses.append(doc["train_losses"])
        ens_val_losses.append(doc["val_losses"])


        model = RNNGATEntropyEstimator(doc["FEATURES_RNN"],
                    doc["FEATURES_GAT"], doc["FEATURES_RHO"],
                    num_samples=doc["NUM_BATCH_SAMPLES"],
                    num_heads=doc["NUM_HEADS"], with_N=doc["WITH_N"],
                    avg_func=jnp.mean)

        with open(DIR + "general/results/" + NAME + INFO + _VERSION + "-params", "rb") as f:
            params = model.init(random.PRNGKey(0), data[0, 0])
            params = flax.serialization.from_bytes(params, f.read())

        ens_params.append(params)

        with open(DIR + "general/results/" + NAME + INFO + _VERSION + "-best_params", "rb") as f:
            params = model.init(random.PRNGKey(0), data[0, 0])
            params = flax.serialization.from_bytes(params, f.read())

        ens_best_params.append(params)

    doc["ens_losses"] = ens_losses
    doc["ens_val_losses"] = ens_val_losses
    return theta, data, labels, val_theta, val_data, val_labels, doc, ens_params, ens_best_params
