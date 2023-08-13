# Sampling-efficient estimation of entanglement entropy through supervised learning

Entanglement is a key property of quantum systems, presenting a clear distinction from the
classical world. The extraction of entanglement measures from experiments, however, comes with
significant challenges, often invoking an infeasible sample complexity even for systems of small scale.
Here we explore a supervised machine learning approach that aims to estimate the entanglement
entropy of systems of up to ten qubits from its classical shadow consisting of few samples. We put a
particular focus on estimating both aleatoric and epistemic uncertainty of the network’s estimate and
benchmark against the best known classical estimation algorithms. For states that are contained
in the training distribution, we observe convergence where the baseline method fails to correctly
estimate the desired entanglement measure given a certain budget of samples. We also explore the
model’s extrapolation capabilities, for which we find mixed results, depending on the application.

## Code

In order to run the code the project directory should have the following structure:

```
DIR/
├─ lib/
├─ data/
├─ results/
```

`data` should contain all data files that the network is being trained on. `results` is where the training 
results will be stored. `lib` contains all the code and can be downloaded from this repository. 

### lib

The `lib` folder contains 2 modules (`models.py` and `load_data.py`), 1 main script and 1 demo script. The demo script
shows how the networks output can be loaded and plotted.
