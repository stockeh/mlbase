import copy
import sys
from typing import List

import mlx.core as mx
from mlx.utils import tree_flatten


def parameters_to_vector(parameters: dict):
    vec = []
    for _, val in tree_flatten(parameters):
        vec.append(val.flatten())
    return mx.concatenate(vec)


def vector_to_parameters(vec: mx.array, parameters: dict):
    i = 0
    for _, val in tree_flatten(parameters):
        n_param = val.size
        val[:] = vec[i : i + n_param].reshape(val.shape)
        i += n_param
    return parameters


class SCG:
    def __init__(self, loss_fn, grad_fn):
        super().__init__()
        self._loss_fn = loss_fn
        self._grad_fn = grad_fn

        self._initialized = False
        self._state = {"step": mx.array(0, mx.uint64)}

        self.sigma0 = 1e-6
        self.betamin = 1e-15
        self.betamax = 1e20

    @property
    def state(self):
        return self._state

    @property
    def step(self):
        return self.state["step"]

    def init(self, parameters: dict, fargs: List = []):
        self.state["w"] = parameters_to_vector(parameters)
        self.state["error_old"] = self._loss_fn(self.state["w"], *fargs)
        self.state["error_now"] = self.state["error_old"]
        self.state["gradnew"] = parameters_to_vector(
            self._grad_fn(self.state["w"], *fargs)
        )
        self.state["ftrace"] = [self.state["error_old"].item()]

        self.state["theta"] = 0
        self.state["kappa"] = 0
        self.state["mu"] = 0

        self.state["gradold"] = self.state["gradnew"]
        self.state["d"] = -self.state["gradnew"]
        self.state["success"] = True
        self.state["nsuccess"] = 0
        self.state["beta"] = 1e-6
        self.state["nvars"] = len(self.state["w"])
        self.state["results"] = ""
        self._initialized = True

    def update(self, model, fargs: List = []):
        if not self._initialized:
            self.init(model.trainable_parameters(), fargs)

        self.compute(fargs)
        self.state["step"] = self.step + 1
        model.update(
            vector_to_parameters(self.state["w"], model.trainable_parameters())
        )

    def compute(self, fargs: List = []):
        if self.state["success"]:
            self.state["mu"] = self.state["d"].T @ self.state["gradnew"]
            if self.state["mu"] >= 0:
                self.state["d"] = -self.state["gradnew"]
                self.state["mu"] = self.state["d"].T @ self.state["gradnew"]
            self.state["kappa"] = self.state["d"].T @ self.state["d"]

            if mx.isnan(self.state["kappa"]):
                print("kappa", self.state["kappa"])

            if self.state["kappa"] < sys.float_info.epsilon:
                self.state["reason"] = "limit on machine precision"
                return True

            sigma = self.sigma0 / mx.sqrt(self.state["kappa"])

            w_smallstep = self.state["w"] + sigma * self.state["d"]
            self._loss_fn(w_smallstep, *fargs)
            g_smallstep = self._grad_fn(w_smallstep, *fargs)

            self.state["theta"] = (
                self.state["d"].T @ (g_smallstep - self.state["gradnew"]) / sigma
            )
            if mx.isnan(self.state["theta"]):
                print(f"theta {self.state['theta']}")

        # Increase effective curvature and evaluate step size alpha.
        delta = self.state["theta"] + self.state["beta"] * self.state["kappa"]
        if mx.isnan(delta):
            print(f"delta is NaN")
        elif delta <= 0:
            delta = self.state["beta"] * self.state["kappa"]
            self.state["beta"] = (
                self.state["beta"] - self.state["theta"] / self.state["kappa"]
            )

        if delta == 0:
            self.state["success"] = False
            self.state["error_now"] = self.state["error_old"]
        else:
            alpha = -self.state["mu"] / delta
            # Calculate the comparison ratio Delta
            wnew = self.state["w"] + alpha * self.state["d"]
            error_new = self._loss_fn(wnew, *fargs)
            Delta = (
                2 * (error_new - self.state["error_old"]) / (alpha * self.state["mu"])
            )
            if not mx.isnan(Delta) and Delta >= 0:
                self.state["success"] = True
                self.state["nsuccess"] += 1
                self.state["w"][:] = wnew
                self.state["error_now"] = error_new
            else:
                self.state["success"] = False
                self.state["error_now"] = self.state["error_old"]

        self.state["ftrace"].append(self.state["error_now"].item())

        if self.state["success"]:

            self.state["error_old"] = error_new
            self.state["gradold"][:] = self.state["gradnew"]
            self.state["gradnew"][:] = self._grad_fn(self.state["w"], *fargs)

            # If the gradient is zero then we are done.
            if mx.all(self.state["gradnew"] == 0):
                self.state["reason"] = "zero gradient"
                return True

        if mx.isnan(Delta) or Delta < 0.25:
            self.state["beta"] = min(4 * self.state["beta"], self.betamax)
        elif Delta > 0.75:
            self.state["beta"] = max(0.5 * self.state["beta"], self.betamin)

        # Update search direction using Polak-Ribiere forself.state['mu']la, or re-start
        # in direction of negative gradient after nparams steps.
        if self.state["nsuccess"] == self.state["nvars"]:
            self.state["d"][:] = -self.state["gradnew"]
            self.state["nsuccess"] = 0
        elif self.state["success"]:
            gamma = (self.state["gradold"] - self.state["gradnew"]).T @ (
                self.state["gradnew"] / self.state["mu"]
            )
            self.state["d"][:] = gamma * self.state["d"] - self.state["gradnew"]

        self.state["reason"] = "not yet converged"
        return False


if __name__ == "__main__":

    def loss_fn(w):
        return (w - 1.5) ** 2

    def grad_fn(w):
        return 2 * (w - 1.5)

    parameters = {"linear": mx.array([-5.5])}

    scg = SCG(loss_fn, grad_fn)
    scg.init(parameters, fargs=[])

    for i in range(1000):
        scg.state["step"] = scg.step + 1
        done = scg.compute(fargs=[])
        if done:
            break

    print(scg.state["w"], scg.state["reason"], scg.step)

    import time

    import mlx.nn as nn
    from dataset import mnist
    from models import Network

    train_data, test_data = mnist(-1)
    n_inputs = next(train_data)["image"].shape[1:]
    train_data.reset()

    kwargs = {"n_inputs": n_inputs, "n_hiddens_list": [5] * 2, "n_outputs": 10}

    model = Network(**kwargs)
    model.summary()

    def accuracy_metric(Y, T):
        return mx.mean(mx.argmax(Y, axis=1) == T)

    def loss_fn(X, T):
        return nn.losses.cross_entropy(model(X), T, reduction="mean")

    def grad_fn(X, T):
        train_step_fn = nn.value_and_grad(model, loss_fn)
        _, grads = train_step_fn(X, T)
        return parameters_to_vector(grads)

    def temp_loss_fn(w, X, T):
        old_w = copy.deepcopy(model.trainable_parameters())
        model.update(vector_to_parameters(w, model.trainable_parameters()))
        loss = loss_fn(X, T)
        model.update(old_w)
        return loss

    def temp_grad_fn(w, X, T):
        old_w = copy.deepcopy(model.trainable_parameters())
        model.update(vector_to_parameters(w, model.trainable_parameters()))
        grads = grad_fn(X, T)
        model.update(old_w)
        return grads

    data = next(train_data)
    X = mx.array(data["image"])
    T = mx.array(data["label"])

    scg = SCG(temp_loss_fn, temp_grad_fn)
    scg.init(model.trainable_parameters(), fargs=[X, T])
    state = [model.state, scg.state]

    tic = time.perf_counter()
    for i in range(300):
        scg.update(model, fargs=[X, T])
        mx.eval(state)
    toc = time.perf_counter()
    print()
    print(
        f"{scg.state['reason']}:",
        f"{scg.state['step'].item()} steps,",
        f"finished in {(toc - tic):.2f}s",
    )

    print(f"{(accuracy_metric(model(X), T) * 100).item():.2f}%")

    import matplotlib.pyplot as plt

    plt.ion()
    plt.plot(scg.state["ftrace"])
    plt.show(block=True)
