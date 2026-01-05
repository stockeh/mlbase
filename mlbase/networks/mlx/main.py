import argparse
import copy
import time
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
import models
import tqdm
from dataset import cifar10, mnist
from mlx.optimizers import Optimizer
from optimizers import SCG, parameters_to_vector, vector_to_parameters

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="Network",
    choices=["MLP", "Network", "ViT"],
    help="model architecture",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="mnist",
    choices=["mnist", "cifar10"],
    help="dataset to use",
)
parser.add_argument("-b", "--batch_size", type=int, default=256, help="batch size")
parser.add_argument("-e", "--epochs", type=int, default=5, help="number of epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--cpu", action="store_true", help="use cpu only")


class Metrics:
    def __init__(self, metrics: List[str] = []):
        self.metrics = metrics
        self.functions = {"acc": self.accuracy, "rmse": self.rmse}

    def __getitem__(self, name):
        try:
            return self.functions[name]
        except KeyError:
            raise NotImplementedError(f"{name=} is not implemented.")

    def __call__(self, Y, T, **kwargs):
        results = {}
        for name in self.metrics:
            results[name] = self[name](Y, T, **kwargs)
        return results

    def accuracy(self, Y, T):
        return mx.mean(mx.argmax(Y, axis=1) == T)

    def rmse(self, Y, T):
        return mx.sqrt(mx.mean((Y - T) ** 2))


class NeuralNetwork:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer | Any,
        classification: bool = True,
        metrics: Optional[List[str]] = [],
    ):
        self.model = model
        self.optimizer = optimizer
        self.classification = classification
        self.metrics = Metrics(metrics)

        self.train_error_trace = []
        self.val_error_trace = []

    def loss_fn(self, X, T):
        fn = nn.losses.cross_entropy if self.classification else nn.losses.mse_loss
        return fn(self.model(X), T, reduction="mean")

    def train(self, train_data, test_data, epochs=5):

        state = [self.model.state, self.optimizer.state]
        loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)

        epoch_bar = tqdm.tqdm(range(epochs), desc="Training", unit="epoch", position=0)

        for epoch in epoch_bar:
            self.model.train()
            train_data.reset()

            losses = []
            all_results = []
            samples_per_sec = []

            train_bar = tqdm.tqdm(
                train_data,
                desc=f"Epoch {epoch+1}/{epochs}",
                leave=False,
                unit=" batch",
                position=1,
            )

            for batch in train_bar:
                X = mx.array(batch["image"])
                T = mx.array(batch["label"])

                tic = time.perf_counter()
                if isinstance(self.optimizer, SCG):
                    self.optimizer.update(self.model, fargs=[X, T])
                    loss = self.optimizer.state["fw"]
                else:
                    loss, grads = loss_and_grad_fn(X, T)
                    self.optimizer.update(self.model, grads)
                results = self.metrics(self.model(X), T)
                mx.eval(state)
                toc = time.perf_counter()

                losses.append(loss.item())
                all_results.append(results)
                throughput = X.shape[0] / (toc - tic)
                samples_per_sec.append(throughput)

                train_bar.set_postfix(
                    loss=f"{losses[-1]:.3f}",
                    **{f"{k}": f"{v.item():.3f}" for k, v in all_results[-1].items()},
                    throughput=f"{throughput:.2f}",
                )

            self.model.eval()
            test_loss, test_results = self.evaluate(test_data)
            self.train_error_trace.append(mx.mean(mx.array(losses)).item())
            self.val_error_trace.append(test_loss)
            test_data.reset()

            epoch_bar.set_postfix(
                train_loss=f"{self.train_error_trace[-1]:.3f}",
                test_loss=f"{self.val_error_trace[-1]:.3f}",
                **{f"test_{k}": f"{v.item():.3f}" for k, v in test_results.items()},
                throughput=f"{mx.mean(mx.array(samples_per_sec)).item():.2f}",
            )

    def evaluate(self, test_data):
        losses, all_results = [], []
        for batch in test_data:
            X = mx.array(batch["image"])
            T = mx.array(batch["label"])
            losses.append(self.loss_fn(X, T).item())
            all_results.append(self.metrics(self.model(X), T))
        losses = mx.mean(mx.array(losses)).item()
        all_results = {
            k: mx.mean(mx.array([r[k] for r in all_results]))
            for k in all_results[0].keys()
        }
        return losses, all_results


def main(args):
    mx.random.seed(args.seed)

    if args.dataset == "mnist":
        train_data, test_data = mnist(args.batch_size)
    elif args.dataset == "cifar10":
        train_data, test_data = cifar10(args.batch_size)
    else:
        raise NotImplementedError(f"{args.dataset=} is not implemented.")
    n_inputs = next(train_data)["image"].shape[1:]
    train_data.reset()

    if args.model == "ViT":
        kwargs = {
            "image_size": n_inputs[:-1],
            "channels": n_inputs[-1],
            "patch_size": 4,
            "num_classes": 10,
            "dim": 128,
            "depth": 4,
            "heads": 8,
            "mlp_dim": 128,
        }
    else:
        kwargs = {
            "n_inputs": n_inputs,
            #   'conv_layers_list': [{'filters': 8, 'kernel_size': 3},
            #                        {'filters': 16, 'kernel_size': 3}],
            "n_hiddens_list": [10] * 2,
            "n_outputs": 10,
        }

    from mlx.optimizers import SGD, Adam

    print("[SCG] training...")
    model = getattr(models, args.model)(**kwargs)
    model.summary()

    def loss_fn(X, T):
        return nn.losses.cross_entropy(model(X), T, reduction="mean")

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    def temp_loss_fn(w, X, T):
        old_w = copy.deepcopy(model.trainable_parameters())
        model.update(vector_to_parameters(w, model.trainable_parameters()))
        loss = loss_fn(X, T)
        model.update(old_w)
        return loss

    def temp_grad_fn(w, X, T):
        old_w = copy.deepcopy(model.trainable_parameters())
        model.update(vector_to_parameters(w, model.trainable_parameters()))
        grads = parameters_to_vector(loss_and_grad_fn(X, T)[1])
        model.update(old_w)
        return grads

    optimizer = SCG(temp_loss_fn, temp_grad_fn)
    net = NeuralNetwork(model, optimizer)  # , metrics=['acc'])
    net.train(train_data, test_data, epochs=args.epochs)
    scg_loss = net.train_error_trace

    print("[SGD] training...")
    model = getattr(models, args.model)(**kwargs)

    optimizer = SGD(learning_rate=args.lr)
    net = NeuralNetwork(model, optimizer)  # , metrics=['acc'])
    net.train(train_data, test_data, epochs=args.epochs)
    sgd_loss = net.train_error_trace

    print("[Adam] training...")
    model = getattr(models, args.model)(**kwargs)

    optimizer = Adam(learning_rate=args.lr)
    net = NeuralNetwork(model, optimizer)  # , metrics=['acc'])
    net.train(train_data, test_data, epochs=args.epochs)
    adam_loss = net.train_error_trace

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for d, c, l in zip(
        [adam_loss, sgd_loss, scg_loss],
        ["royalblue", "seagreen", "red"],
        ["Adam", "SGD", "SCG"],
    ):
        ax.plot(mx.exp(-mx.array(d)), color=c, lw=2, label=f"{l}")
    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("Likelihood", fontsize=12)
    ax.set_title(
        f"[{args.dataset.upper()}] bs={args.batch_size} "
        f"-- {args.model} {kwargs['n_hiddens_list']
                                    if 'n_hiddens_list' in kwargs else ''}",
        fontsize=12,
    )
    ax.legend(fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=10)
    fig.savefig(f"media/main_{args.dataset}.png", dpi=300, bbox_inches="tight")

    plt.show(block=True)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.cpu:
        mx.set_default_device(mx.cpu)
    main(args)
