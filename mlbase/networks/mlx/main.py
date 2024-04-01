import time
import tqdm
import argparse

import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim

from functools import partial
from mlx.optimizers import Optimizer
from typing import List, Optional

import models
from dataset import mnist, cifar10

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('--arch', type=str,
                    default='ViT', choices=['MLP', 'Network', 'ViT'],
                    help='model architecture')
parser.add_argument('--dataset', type=str,
                    default='mnist', choices=['mnist', 'cifar10'],
                    help='dataset to use')
parser.add_argument('-b', '--batch_size', type=int,
                    default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--cpu', action='store_true', help='use cpu only')


class Metrics:
    def __init__(self, metrics: List[str] = []):
        self.metrics = metrics
        self.functions = {
            'acc': self.accuracy_metric,
            'rmse': self.rmse_metric
        }

    def __getitem__(self, name):
        try:
            return self.functions[name]
        except KeyError:
            raise NotImplementedError(f'{name=} is not implemented.')

    def __call__(self, Y, T, **kwargs):
        results = {}
        for name in self.metrics:
            results[name] = self[name](Y, T, **kwargs)
        return results

    def accuracy_metric(self, Y, T):
        return mx.mean(mx.argmax(Y, axis=1) == T)

    def rmse_metric(self, Y, T):
        return mx.sqrt(mx.mean((Y - T) ** 2))


class NeuralNetwork:
    def __init__(self, model, optimizer: Optional[Optimizer] = None,
                 metrics: Optional[List[str]] = []):
        self.model = model
        self.optimizer = optimizer if optimizer else optim.Adam(1e-3)
        self.metrics = Metrics(metrics)

        self.train_error_trace = []
        self.val_error_trace = []

    def eval_fn(self, X, T):
        Y = self.model(X)
        loss = nn.losses.cross_entropy(Y, T, reduction='mean')
        results = self.metrics(Y, T)
        return loss, results

    def train(self, train_data, test_data, epochs=5):

        state = [self.model.state, self.optimizer.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def step(X, T):
            train_step_fn = nn.value_and_grad(self.model, self.eval_fn)
            (loss, results), grads = train_step_fn(X, T)
            self.optimizer.update(self.model, grads)
            return loss, results

        epoch_bar = tqdm.tqdm(range(epochs), desc='Training',
                              unit='epoch', position=0)
        for epoch in epoch_bar:
            self.model.train()
            train_data.reset()

            losses = []
            all_results = []
            samples_per_sec = []

            train_bar = tqdm.tqdm(
                train_data, desc=f'Epoch {epoch+1}/{epochs}',
                leave=False, unit=' batch', position=1
            )

            for batch in train_bar:
                X = mx.array(batch['image'])
                T = mx.array(batch['label'])

                tic = time.perf_counter()
                loss, results = step(X, T)
                mx.eval(state)
                toc = time.perf_counter()

                losses.append(loss.item())
                all_results.append(results)
                throughput = X.shape[0] / (toc - tic)
                samples_per_sec.append(throughput)

                train_bar.set_postfix(
                    loss=f'{losses[-1]:.3f}',
                    **{f'{k}': f'{v.item():.3f}' for k, v in all_results[-1].items()},
                    throughput=f'{throughput:.2f}'
                )

            self.model.eval()
            test_loss, test_results = self.evaluate(test_data)
            test_data.reset()

            epoch_bar.set_postfix(
                train_loss=f'{mx.mean(mx.array(losses)).item():.3f}',
                test_loss=f'{test_loss.item():.3f}',
                **{f'test_{k}': f'{v.item():.3f}' for k, v in test_results.items()},
                throughput=f'{mx.mean(mx.array(samples_per_sec)).item():.2f}'
            )

    def evaluate(self, test_data):
        losses, all_results = [], []
        for batch in test_data:
            X = mx.array(batch['image'])
            T = mx.array(batch['label'])
            loss, results = self.eval_fn(X, T)
            losses.append(loss.item())
            all_results.append(results)
        losses = mx.mean(mx.array(losses))
        all_results = {k: mx.mean(
            mx.array([r[k] for r in all_results])) for k in all_results[0].keys()}
        return losses, all_results


def main(args):
    mx.random.seed(args.seed)

    if args.dataset == 'mnist':
        train_data, test_data = mnist(args.batch_size)
    elif args.dataset == 'cifar10':
        train_data, test_data = cifar10(args.batch_size)
    else:
        raise NotImplementedError(f'{args.dataset=} is not implemented.')
    n_inputs = next(train_data)['image'].shape[1:]
    train_data.reset()

    if args.arch == 'ViT':
        kwargs = {'image_size': n_inputs[:-1],
                  'channels': n_inputs[-1],
                  'patch_size': 4,
                  'num_classes': 10,
                  'dim': 128,
                  'depth': 4,
                  'heads': 8,
                  'mlp_dim': 128
                  }
    else:
        kwargs = {'n_inputs': n_inputs,
                  'conv_layers_list': [{'filters': 8, 'kernel_size': 3},
                                       {'filters': 16, 'kernel_size': 3}],
                  'n_hiddens_list': [10]*1,
                  'n_outputs': 10
                  }

    model = getattr(models, args.arch)(**kwargs)
    model.summary()

    optimizer = optim.Adam(learning_rate=args.lr)
    net = NeuralNetwork(model, optimizer, metrics=['acc'])
    net.train(train_data, test_data, epochs=args.epochs)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.cpu:
        mx.set_default_device(mx.cpu)
    main(args)
