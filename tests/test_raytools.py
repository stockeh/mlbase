# need to add module path for [raytools, neuralnetworks, etc]
# to PYTHONPATH in bashrc to avoid a ModuleNotFoundError
# export PYTHONPATH="${PYTHONPATH}:/s/chopin/l/grad/stock/research/mlbase"
# try:
#     import sys
#     PYTHONPATH = '/s/chopin/l/grad/stock/research/mlbase'
#     sys.path.append(PYTHONPATH)
# except:
#     raise

import gzip
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import ray

from mlbase.networks.pytorch import neuralnetworks as nn
from mlbase.utilities import mlutilities as ml
from mlbase.utilities import raytools


def load_data():
    with gzip.open("/s/chopin/d/proj/ml/users/stock/data/mnist.pkl.gz", "rb") as f:
        train_set, _, test_set = pickle.load(f, encoding="latin1")

    Xtrain = train_set[0].reshape((-1, 1, 28, 28))
    Ttrain = train_set[1].reshape((-1, 1))

    Xtest = test_set[0].reshape((-1, 1, 28, 28))
    Ttest = test_set[1].reshape((-1, 1))

    return Xtrain, Ttrain, Xtest, Ttest


def main():
    primary = os.uname()[1]
    port = 6813
    raytools.ray_start(primary, port, ["lamborghini", "porsche"])
    #    pythonpath=PYTHONPATH)

    @ray.remote(num_gpus=1)
    def run(hiddens):
        # need to load data in ray.remote to reduce remote footprint
        Xtrain, Ttrain, Xtest, Ttest = load_data()
        host = os.uname()[1]
        print(host, hiddens)
        # train network
        nnet = nn.NeuralNetworkClassifier(
            Xtrain.shape[1:], hiddens, len(np.unique(Ttrain)), use_gpu=True, seed=1234
        )
        nnet.train(
            Xtrain,
            Ttrain,
            n_epochs=100,
            batch_size=128,
            learning_rate=0.001,
            opt="accsgd",
            validation_data=(Xtest, Ttest),
            verbose=False,
        )
        # evaluate
        results = [
            host,
            hiddens,
            ml.percent_correct(Ttrain, nnet.use(Xtrain)),
            ml.percent_correct(Ttest, nnet.use(Xtest)),
        ]
        return results, nnet

    start_t = time.time()
    results = ray.get([run.remote([u] * l) for u in [10, 20] for l in range(1, 5)])
    print(f"Finished in {time.time() - start_t:.3f} seconds.")
    raytools.ray_stop()

    print([r[0] for r in results])
    top_i = np.argmax([r[0][-1] for r in results])
    nnet = results[top_i][1]

    # plotting
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(nnet.train_error_trace, label="train")
    ax.plot(nnet.val_error_trace, label="val")
    ax.set_title(f"{nnet.model.n_hiddens_list} {results[top_i][0][-1]:.3f}%")

    plt.show()


if __name__ == "__main__":
    main()
