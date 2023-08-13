import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
from typing import Sequence, Callable


class GATLayer(nn.Module):
    """
    Single GAT layer for FULLY CONNECTED graphs.
    Acts on NxF 2D matrices where N is the number of nodes (e.g. number of spins)
    and F is the number of input features per node (does not need to be
    specified).
    self.features is the number of output features however and needs to
    be specified.
    """
    features: int
    N: int

    def setup(self):
        self.W = nn.Dense(features=self.features, use_bias=False)
        self.a = nn.Dense(features=1, use_bias=False)

    def __call__(self, graph):
        ltg = self.W(graph)

        def attention(i, j):
            """simple attention function proposed in the GAT paper.
            Can be swapped out by arbitrary attention functions with the
            same signature"""
            e_ij = self.a(jnp.concatenate((ltg[i], ltg[j]), -1))
            return jax.nn.leaky_relu(e_ij, negative_slope=0.2)

        # builds matrix containing all attention coefficients
        E = jnp.fromfunction(attention, shape=(self.N, self.N),
                            dtype="i1").squeeze()

        Alpha = nn.softmax(E)
        #H = jnp.einsum("jk,ij->ik", ltg, Alpha)
        H = jnp.dot(Alpha, ltg)

        return H

class GATLayerMultihead(nn.Module):
    """
    Apply multiple 'GATLayers' with different parameters in parallel (heads)
    For self.heads = 1 this is effectively the same as the 'GATLayer' class
    """
    features: int
    N: int
    heads: int = 1

    """def setup(self):
        self.layers = [GATLayer(self.features, self.N, name="head"+str(i))
                                 for i in range(1, self.heads + 1)]
    """
    @nn.compact
    def __call__(self, graph):
        # return list that stores the outputs of each head
        vGATLayer = nn.vmap(GATLayer, in_axes=0, out_axes=0,
                            variable_axes={'params': 0},
                            split_rngs={'params': True})
        Hs = vGATLayer(self.features, self.N)(jnp.tile(graph, (self.heads, 1)).reshape(self.heads, graph.shape[0], graph.shape[1]))
        return Hs

class GATStack(nn.Module):
    """
    Stack of 'GATLayerMultihead' layers.
    Specify self.features to determine the size of the stack and the output
    features per layer in the stack.
    """
    features: Sequence[int]
    N: int
    heads: int = 1

    def setup(self):
        self.GAT_layers = [GATLayerMultihead(feat, self.N, self.heads)
                           for feat in self.features]

    def __call__(self, x):
        for (i, lyr) in enumerate(self.GAT_layers):
            x = lyr(x)
            if i != len(self.GAT_layers) - 1:
                # the independent results for each head are being concatenated
                x = jnp.concatenate(x, axis=1)
                x = jax.nn.leaky_relu(x, negative_slope=0.2)
            else:
                # for the last layer the results for each head are being averaged
                x = jnp.mean(jnp.array(x), axis=0)
        return x


class RNN(nn.Module):
    """
    Ordinary LSTM cell.
    """
    @staticmethod
    def initialize_carry(num_inputs):
        carry = nn.OptimizedLSTMCell.initialize_carry(random.PRNGKey(0), (), num_inputs)
        return carry

    @nn.compact
    def __call__(self, carry, xs):
        cell_scan = nn.scan(nn.OptimizedLSTMCell, variable_broadcast="params",
                            split_rngs={"params": False})
        res = cell_scan()(carry, xs)
        return res[1]


class RNNStack(nn.Module):
    """
    Stack of 'RNN' cells.
    Specify self.features to determine the size of the stack and the output
    features per layer in the stack. 
    """
    features: Sequence[int]

    def setup(self):
        self.layers = [RNN() for feat in self.features]

    def __call__(self, xs):
        N = xs[-1, 0].astype("i4")
        for (feat, lyr) in zip(self.features, self.layers):
            c = lyr.initialize_carry(feat)
            xs = lyr(c, xs)
        return xs[N-1]


class NN(nn.Module):
    """
    Simple dense feed forward neural network.
    self.features specifies the number of layers and number of features per
    layer.
    """
    features: Sequence[int]

    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features]

    def __call__(self, x):
        x = x
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = nn.elu(x)
        return jax.nn.softplus(x)


class RNNGATEntropyEstimator(nn.Module):
    """
    Model to estimate the entropy of the state from samples of its POVM
    distribution.

    self.features_gat, self.features_rnn and self.features_rho: specify the number of layers and
    features for the respective subnetworks

    self.num_samples: number of nodes (e.g. number of samples per batch)

    self.heads: number of heads to be computed for each GAT layer

    self.avg_function: function that is used to obtain perm. inv. quantity from graph nodes 
    """
    features_rnn: Sequence[int]
    features_gat: Sequence[int]
    features_rho: Sequence[int]
    num_samples: int
    num_heads: int = 1
    avg_func: Callable = jnp.mean

    def setup(self):
        self.rho = NN(self.features_rho)
        self.rnn = RNNStack(self.features_rnn)
        self.gat = GATStack(self.features_gat, self.num_samples, self.num_heads)

    def __call__(self, x):
        vrnn = vmap(self.rnn, in_axes=0)
        group_mean = self.avg_func(self.gat(nn.elu(vrnn(x))), axis=0)
        # add small offset to ensure sigma != 0
        return self.rho(group_mean).squeeze() + 1e-6
