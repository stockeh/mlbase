import matplotlib.pyplot as plt
import numpy as np

from mlbase.networks.python import neuralnetworks as pynn
from mlbase.networks.pytorch import neuralnetworks as torchnn
from mlbase.networks.tensorflow import neuralnetworks as tfnn
from mlbase.utilities import mlutilities as ml

if __name__ == "__main__":
    plt.ion()
    br = "".join(["-"] * 8)

    np.random.seed(42)
    X = np.arange(100).reshape((-1, 1))
    T = np.sin(X * 0.04)

    n_epochs = 1000
    n_hiddens_list = [5] * 2
    learning_rate = 0.001
    optimizer = "adam"
    activation = "tanh"

    print(f"{br}Testing Python NeuralNetwork{br}")
    # ---------------------------------------------------------------#

    nnet = pynn.NeuralNetwork(
        X.shape[1], n_hiddens_list, T.shape[1], activation=activation
    )
    nnet.train(
        X, T, n_epochs, method=optimizer, learning_rate=learning_rate, verbose=True
    )
    Y = nnet.use(X)
    print(f"RMSE {ml.rmse(Y, T):.3f} in {nnet.training_time:.3f} s")

    plt.figure(1)
    plt.plot(nnet.error_trace**2)  # objective to actual
    plt.title("Python Network")
    plt.figure(2)
    nnet.draw()
    plt.title("Python Network")

    print(f"{br}Testing PyTorch NeuralNetwork{br}")
    # ---------------------------------------------------------------#

    nnet = torchnn.NeuralNetwork(
        X.shape[1], n_hiddens_list, T.shape[1], activation_f=activation
    )
    nnet.summary()
    nnet.train(
        X, T, n_epochs, batch_size=-1, learning_rate=learning_rate, opt=optimizer
    )
    Y = nnet.use(X)
    print(f"RMSE {ml.rmse(Y, T):.3f} in {nnet.training_time:.3f} s")

    plt.figure(3)
    plt.plot(nnet.train_error_trace)
    plt.title("PyTorch Network")

    print(f"{br}Testing TensorFlow NeuralNetwork{br}")
    # ---------------------------------------------------------------#

    nnet = tfnn.NeuralNetwork(
        X.shape[1], n_hiddens_list, T.shape[1], activation=activation
    )

    nnet.train(
        X,
        T,
        n_epochs,
        batch_size=X.shape[0],
        learning_rate=learning_rate,
        optimizer=optimizer,
    )

    Y = nnet.use(X)
    print(f"RMSE {ml.rmse(Y, T):.3f} in {nnet.training_time:.3f} s")

    plt.figure(4)
    plt.plot(nnet.history["loss"])
    plt.title("TensorFlow Network")

    plt.show(block=True)
