import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.animation import FuncAnimation
from torch.optim import Adam, SGD
from tqdm import tqdm

from mlbase.networks.pytorch.optimizers.scg_optim_pytorch import SCG

if __name__ == '__main__':

    test_rosenbrock = True

    ######################################################################
    if test_rosenbrock:
        # Example is adapted from mildlyoverfitted code and tutorial
        # https://github.com/jankrepl/mildlyoverfitted
        def rosenbrock(xy):
            x, y = xy
            return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

        def run_optimization(xy_init, optimizer_class, n_iter, **optimizer_kwargs):
            xy_t = torch.tensor(xy_init, requires_grad=True)
            optimizer = optimizer_class([xy_t], **optimizer_kwargs)

            path = np.empty((n_iter + 1, 2))
            path[0, :] = xy_init

            def closure(grad=True, loss=None):
                '''Only for SCG'''
                if not loss:
                    loss = rosenbrock(xy_t)
                if not grad:
                    return loss
                optimizer.zero_grad()
                loss.backward()
                return loss

            for i in tqdm(range(1, n_iter + 1)):

                if isinstance(optimizer, SCG):
                    optimizer.step(closure)

                else:
                    optimizer.zero_grad()
                    loss = rosenbrock(xy_t)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(xy_t, 1.0)
                    optimizer.step()

                path[i, :] = xy_t.detach().numpy()

            return path

        def create_animation(paths,
                             colors,
                             names,
                             figsize=(12, 12),
                             x_lim=(-2, 2),
                             y_lim=(-1, 3),
                             n_seconds=5):
            if not (len(paths) == len(colors) == len(names)):
                raise ValueError

            path_length = max(len(path) for path in paths)

            n_points = 300
            x = np.linspace(*x_lim, n_points)
            y = np.linspace(*y_lim, n_points)
            X, Y = np.meshgrid(x, y)
            Z = rosenbrock([X, Y])

            minimum = (1.0, 1.0)

            fig, ax = plt.subplots(figsize=figsize)
            ax.contour(X, Y, Z, 90, cmap='jet')

            lines = [ax.plot([], [], '.-',
                             label=label,
                             c=c) for c, label in zip(colors, names)]

            ax.legend(prop={'size': 25})
            ax.plot(*minimum, 'rD')

            def animate(i):
                for path, line in zip(paths, lines):
                    # set_offsets(path[:i, :])
                    line[0].set_xdata(path[:i+1, 0])
                    # set_offsets(path[:i, :])
                    line[0].set_ydata(path[:i+1, 1])

                ax.set_title(str(i))

            ms_per_frame = 1000 * n_seconds / path_length

            anim = FuncAnimation(
                fig, animate, frames=path_length, interval=ms_per_frame)
            return anim

        xy_init = (.3, .8)
        n_iter = 200

        path_adam = run_optimization(xy_init, Adam, n_iter, lr=0.01)
        path_sgd = run_optimization(xy_init, SGD, n_iter, lr=0.01)
        path_scg = run_optimization(xy_init, SCG, n_iter)

        freq = 1

        paths = [path_adam[::freq], path_sgd[::freq], path_scg[::freq]]
        colors = ['green', 'blue', 'black']
        names = ['Adam', 'SGD', 'SCG']

        anim = create_animation(paths,
                                colors,
                                names,
                                figsize=(12, 7),
                                x_lim=(-.1, 1.1),
                                y_lim=(-.1, 1.1),
                                n_seconds=7)

        print('sgd')
        print(path_sgd[-15:])
        print('adam')
        print(path_adam[-15:])
        print('scg')
        print(path_scg[-15:])

        print('Creating animation ...')
        fname = 'media/result_scg.gif'
        anim.save(fname)

        print(f'Resulting animation is in {fname}')
