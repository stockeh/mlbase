import copy
import random
import time
from numbers import Integral

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class TrainLogger(tf.keras.callbacks.Callback):
    """Helper class to log metrics."""

    def __init__(self, n_epochs, step=10):
        self.step = step

        self.n_epochs = n_epochs

    def on_epoch_end(self, epoch, logs=None):
        s = f"epoch: {epoch}, rmse {logs['root_mean_squared_error']:7.5f}"
        if "val_root_mean_squared_error" in logs:
            s += f", val_rmse {logs['val_root_mean_squared_error']:7.5f}"
        if epoch % self.step == 0:
            print(s)
        elif epoch + 1 == self.n_epochs:
            print(s)
            print("finished!")


class NeuralNetwork:
    """NeuralNetwork class to init, train, and use TensorFlow models.

    Three primary models that can be created include:

      - Linear Regression Models,
      - Fully-Connected Neural Networks, and
      - Convolutional Neural networks.

    This class handles initalizing the models, standardizing the data,
    training the model, and using the model (with un/standardizing).
    """

    def __init__(
        self,
        n_inputs,
        n_hiddens_list,
        n_outputs,
        conv_layers=[],
        activation="tanh",
        seed=None,
    ):
        """Initialize the model according to arguments

        :param n_inputs: a shape tuple (integers), not including the
            batch size. e.g. (128,128,1) or (32,). A plain integer is
            also accepted and is treated as a one-dimensional shape.
        :param n_hiddens_list: list (integers) specifying the number of
            hidden units. e.g. [10, 10] two layers with 10 units each or
            [0] to have no hidden layers
        :param n_outputs: a shape tuple (integers), not including the
            batch size. e.g. (10,) or (1,)
        :param conv_layers: list (dicts) specifying the number of
            2d-conv layers with `n_units` and `shape` keys. e.g.
            [{'n_units': 8, 'shape': [3, 3]},
             {'n_units': 8, 'shape': [3, 3]}]
            specifies two convolutional layers with 8 3x3 filters each
        :param activation: activation function following conv or dense
        :param seed: random seed used for reproducibility
        """

        if not isinstance(n_hiddens_list, list):
            raise Exception(f"{type(self).__name__}: n_hiddens_list must be a list.")

        self.seed = seed
        self._set_seed()
        tf.keras.backend.clear_session()

        self.n_inputs = self._normalize_shape(n_inputs, "n_inputs")
        self.conv_layers = conv_layers
        self.n_hiddens_list = n_hiddens_list
        self.n_outputs = n_outputs

        X = Z = tf.keras.Input(shape=self.n_inputs)

        # -----------------------
        # convolutional layers
        if conv_layers:
            for conv in conv_layers:
                Z = tf.keras.layers.Conv2D(
                    conv["n_units"],
                    kernel_size=conv["shape"],
                    strides=1,
                    padding="same",
                )(Z)
                Z = tf.keras.layers.Activation(activation)(Z)
                Z = tf.keras.layers.MaxPooling2D(pool_size=2)(Z)

        # -----------------------
        # fully-connected layers
        Z = tf.keras.layers.Flatten()(Z)
        if not (n_hiddens_list == [] or n_hiddens_list == [0]):
            for i, units in enumerate(n_hiddens_list):
                Z = tf.keras.layers.Dense(units)(Z)
                Z = tf.keras.layers.Activation(activation)(Z)
        # -----------------------
        # linear output layer
        Y = tf.keras.layers.Dense(n_outputs, name="out")(Z)

        self.model = tf.keras.Model(inputs=X, outputs=Y)

        # Member variables for standardization
        self.standardize_x = True
        self.standardize_t = True
        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None

        self.history = None
        self.training_time = None

    def __repr__(self):
        str = f"{type(self).__name__}({self.n_inputs}, {self.n_hiddens_list}, {self.n_outputs})"
        if self.history:
            str += f"\n  Final objective value is {self.history['loss'][-1]:.5f} in {self.training_time:.4f} seconds."
        else:
            str += "  Network is not trained."
        return str

    def summary(self):
        return self.model.summary()

    def _set_seed(self):
        if self.seed:
            np.random.seed(self.seed)
            random.seed(self.seed)
            tf.random.set_seed(self.seed)

    def _standardizeX(self, X):
        if self.standardize_x:
            result = (X - self.Xmeans) / self.XstdsFixed
            result[:, self.Xconstant] = 0.0
            return result
        else:
            return X

    def _unstandardizeX(self, Xs):
        return (self.Xstds * Xs + self.Xmeans) if self.standardize_x else Xs

    def _standardizeT(self, T):
        if self.standardize_t:
            result = (T - self.Tmeans) / self.TstdsFixed
            result[:, self.Tconstant] = 0.0
            return result
        else:
            return T

    def _unstandardizeT(self, Ts):
        return (self.Tstds * Ts + self.Tmeans) if self.standardize_t else Ts

    def _setup_standardize(self, X, T):
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
            self.Xconstant = self.Xstds == 0
            self.XstdsFixed = copy.copy(self.Xstds)
            self.XstdsFixed[self.Xconstant] = 1

        if self.Tmeans is None:
            self.Tmeans = T.mean(axis=0)
            self.Tstds = T.std(axis=0)
            self.Tconstant = self.Tstds == 0
            self.TstdsFixed = copy.copy(self.Tstds)
            self.TstdsFixed[self.Tconstant] = 1

    def train(
        self,
        X,
        T,
        n_epochs,
        batch_size,
        optimizer="sgd",
        learning_rate=0.001,
        loss_f="mse",
        validation=None,
        standardize_x=True,
        standardize_t=True,
        shuffle=False,
        verbose=False,
    ):
        """Use Keras Functional API to train nnet

        :param X: numpy array of images (N,W,H,C)
        :param T: numpy array of target values (N,F)
        :param n_epochs: number of epochs to train for
        :param batch_size: size of batch (integer) used in each epoch
        :param optimizer: name of optimization function used to train
        :param learning_rate: effective step size (float) used in training
        :param loss_f: loss function to use for training
        :param validation: tuple of (X, T) values used for validation
        :param standardize_x: boolean if X should be standardized or not
        :param standardize_t: boolean if T should be standardized or not
        :param verbose: boolean if metrics should be printed during training
        """

        self._set_seed()
        self.batch_size = batch_size
        self.standardize_x = standardize_x
        self.standardize_t = standardize_t

        self._setup_standardize(X, T)
        T = self._standardizeT(T)
        X = self._standardizeX(X)

        validation_data = None
        if validation:
            assert (
                len(validation) == 2
            ), "validation must be of the following shape: (X, T)"
            validation_data = (
                self._standardizeX(validation[0]),
                self._standardizeT(validation[1]),
            )

        try:
            if optimizer.lower() == "sgd":  # default
                optim = tf.keras.optimizers.SGD(learning_rate=learning_rate)
            elif optimizer.lower() == "adam":
                optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        except:
            raise Exception(
                f"train: optimizer={optimizer} has not yet been implemented."
            )

        try:
            if loss_f.lower() == "mse":  # default
                loss = tf.keras.losses.MSE
            elif loss_f.lower() == "mae":
                loss = tf.keras.losses.MAE
        except:
            raise Exception(f"train: loss_f={loss_f} has not yet been implemented.")

        callback = None
        if verbose:
            callback = [TrainLogger(n_epochs, step=n_epochs // 5)]

        self.model.compile(
            optimizer=optim,
            loss=loss,
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(),
                tf.keras.metrics.MeanSquaredError(),
                tf.keras.metrics.MeanAbsoluteError(),
            ],
        )

        start_time = time.time()
        self.history = self.model.fit(
            X,
            T,
            batch_size=batch_size,
            epochs=n_epochs,
            shuffle=shuffle,
            verbose=0,
            callbacks=callback,
            validation_data=validation_data,
        ).history
        self.training_time = time.time() - start_time

        return self

    def use(self, X):
        """Use the trained TensorFlow model.

        :param X: numpy array of images (N,W,H,C)
        :param Y: numpy array of predicted values (N,F)
        """
        # Set to error logging after model is trained
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        Y = self._unstandardizeT(
            self.model.predict(self._standardizeX(X), batch_size=self.batch_size)
        )
        return Y

    @staticmethod
    def _normalize_shape(shape, name):
        """Ensure Keras receives a tuple shape even if caller passes an int."""
        if isinstance(shape, Integral):
            if shape <= 0:
                raise ValueError(f"{name} must be a positive integer.")
            return (int(shape),)

        try:
            normalized = tuple(NeuralNetwork._normalize_dim(dim, name) for dim in shape)
        except TypeError as exc:
            raise ValueError(
                f"{name} must be an int or iterable of ints (optionally None)."
            ) from exc

        if len(normalized) == 0:
            raise ValueError(f"{name} must describe at least one dimension.")

        return normalized

    @staticmethod
    def _normalize_dim(dim, name):
        if dim is None:
            return None
        if isinstance(dim, Integral):
            dim = int(dim)
            if dim <= 0:
                raise ValueError(f"{name} dimensions must be positive; got {dim}.")
            return dim
        raise ValueError(f"{name} dimensions must be integers or None; got {dim}.")


if __name__ == "__main__":
    print("Testing NeuralNetwork for regression,", end=" ")
    # ---------------------------------------------------------------#
    X = np.linspace(-1, 1, 100).reshape((-1, 1))
    T = np.sin(X * np.pi)

    n_hiddens_list = [10, 10]  # two layer network with 10 units each

    nnet = NeuralNetwork(X.shape[1], n_hiddens_list, T.shape[1], activation="tanh")

    nnet.train(X, T, n_epochs=800, batch_size=32, learning_rate=0.001, optimizer="adam")

    Y = nnet.use(X)

    print(f"RMSE: {np.sqrt(np.mean((Y - T)**2)):.3f}")
    plt.plot(nnet.history["loss"])
    plt.show()
