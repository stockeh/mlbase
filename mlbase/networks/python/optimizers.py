import numpy as np
import sys
import time
import math  # for math.ceil

######################################################################
# sgd


def sgd(w, error_f, fargs=[], n_iterations=100, error_gradient_f=None,
        eval_f=lambda x: x, save_wtrace=False, verbose=False,
        learning_rate=0.001, momentum_rate=0.0):

    start_time = time.time()
    start_time_last_verbose = start_time

    w = w.copy()
    wtrace = [w.copy()] if save_wtrace else None

    ftrace = [eval_f(error_f(w, *fargs))]

    w_change = 0

    for iteration in range(n_iterations):

        error = error_f(w, *fargs)
        grad = error_gradient_f(w, *fargs)
        w_change = -learning_rate * grad + momentum_rate * w_change
        w += w_change
        if save_wtrace:
            wtrace.append(w.copy())
        ftrace.append(eval_f(error))

        iterations_per_print = math.ceil(n_iterations/10)
        if verbose and (iteration + 1) % max(1, iterations_per_print) == 0:
            seconds = time.time() - start_time_last_verbose
            eval = eval_f(error)
            print(
                f'sgd: Iteration {iteration+1:d} ObjectiveF={eval:.5f} Seconds={seconds:.3f}')
            start_time_last_verbose = time.time()

    return {'w': w,
            'f': error_f(w, *fargs),
            'n_iterations': iteration,
            'wtrace': np.array(wtrace)[:iteration + 2, :] if save_wtrace else None,
            'ftrace': np.array(ftrace)[:iteration + 2],
            'reason': 'iterations',
            'time': time.time() - start_time}


######################################################################
# adam

def adam(w, error_f, fargs=[], n_iterations=100, error_gradient_f=None,
         eval_f=lambda x: x, save_wtrace=False, verbose=False,
         learning_rate=0.001, momentum_rate=None):

    start_time = time.time()
    start_time_last_verbose = start_time

    w = w.copy()
    wtrace = [w.copy()] if save_wtrace else None

    beta1 = 0.9
    beta2 = 0.999
    alpha = learning_rate
    epsilon = 10e-8
    nW = len(w)
    g = np.zeros((nW))
    g2 = np.zeros((nW))
    beta1t = beta1
    beta2t = beta2

    ftrace = [eval_f(error_f(w, *fargs))]

    for iteration in range(n_iterations):

        error = error_f(w, *fargs)

        grad = error_gradient_f(w, *fargs)
        # aproximate first and second moment
        g = beta1 * g + (1 - beta1) * grad
        g2 = beta2 * g2 + (1 - beta2) * grad * grad
        # bias corrected moment estimates
        g_corrected = g / (1 - beta1t)
        g2_corrected = g2 / (1 - beta2t)
        w -= alpha * g_corrected / (np.sqrt(g2_corrected) + epsilon)
        if save_wtrace:
            wtrace.append(w.copy())
        ftrace.append(eval_f(error))

        beta1t *= beta1
        beta2t *= beta2

        iterations_per_print = math.ceil(n_iterations/10)
        if verbose and (iteration + 1) % max(1, iterations_per_print) == 0:
            seconds = time.time() - start_time_last_verbose
            eval = eval_f(error)
            print(
                f'adam: Iteration {iteration+1:d} ObjectiveF={eval:.5f} Seconds={seconds:.3f}')
            start_time_last_verbose = time.time()

    return {'w': w,
            'f': error_f(w, *fargs),
            'n_iterations': iteration,
            'wtrace': np.array(wtrace)[:iteration + 2, :] if save_wtrace else None,
            'ftrace': np.array(ftrace)[:iteration + 2],
            'reason': 'iterations',
            'time': time.time() - start_time}

######################################################################
# Scaled Conjugate Gradient algorithm from
#  "A Scaled Conjugate Gradient Algorithm for Fast Supervised Learning"
#  by Martin F. Moller
#  Neural Networks, vol. 6, pp. 525-533, 1993
#
#  Adapted by Chuck Anderson from the Matlab implementation by Nabney
#   as part of the netlab library.
#


def scg(w, error_f, fargs=[], n_iterations=100, error_gradient_f=None,
        eval_f=lambda x: x, save_wtrace=False, verbose=False,
        learning_rate=None, momentum_rate=None):  # not used here

    float_precision = sys.float_info.epsilon

    start_time = time.time()
    start_time_last_verbose = start_time

    w = w.copy()
    wtrace = [w.copy()] if save_wtrace else None
    isnan = np.isnan
    sqrt = math.sqrt

    sigma0 = 1.0e-6
    error_old = error_f(w, *fargs)
    error_now = error_old
    gradnew = error_gradient_f(w, *fargs)
    ftrace = [eval_f(error_old)]

    gradold = gradnew
    d = -gradnew       # Initial search direction.
    success = True     # Force calculation of directional derivs.
    nsuccess = 0       # nsuccess counts number of successes.
    beta = 1.0e-6      # Initial scale parameter. Lambda in Moeller.
    betamin = 1.0e-15  # Lower bound on scale.
    betamax = 1.0e20   # Upper bound on scale.
    nvars = len(w)
    iteration = 1      # count of number of iterations

    thisIteration = 1
    while thisIteration <= n_iterations:

        if success:
            mu = d.T @ gradnew
            if mu >= 0:
                d = -gradnew
                mu = d.T @ gradnew
            kappa = d.T @ d

            if isnan(kappa):
                print('kappa', kappa)

            if kappa < float_precision:
                return {'w': w,
                        'f': error_now,
                        'n_iterations': iteration,
                        'wtrace': np.array(wtrace)[:iteration + 1, :] if save_wtrace else None,
                        'ftrace': np.array(ftrace)[:iteration + 1],
                        'reason': 'limit on machine precision',
                        'time': time.time() - start_time}
            sigma = sigma0 / sqrt(kappa)

            w_smallstep = w + sigma * d
            error_f(w_smallstep, *fargs)
            g_smallstep = error_gradient_f(w_smallstep, *fargs)

            theta = d.T @ (g_smallstep - gradnew) / sigma
            if isnan(theta):
                print(
                    f'theta {theta} sigma {sigma} d[0] {d[0]} g_smallstep[0] {g_smallstep[0]} gradnew[0] {gradnew[0]}')

        # Increase effective curvature and evaluate step size alpha.
        delta = theta + beta * kappa
        if isnan(delta):
            print(f'delta is NaN theta {theta} beta {beta} kappa {kappa}')
        elif delta <= 0:
            delta = beta * kappa
            beta = beta - theta / kappa

        if delta == 0:
            success = False
            error_now = error_old
        else:
            alpha = -mu / delta
            # Calculate the comparison ratio Delta
            wnew = w + alpha * d
            error_new = error_f(wnew, *fargs)
            Delta = 2 * (error_new - error_old) / (alpha * mu)
            if not isnan(Delta) and Delta >= 0:
                success = True
                nsuccess += 1
                w[:] = wnew
                error_now = error_new
            else:
                success = False
                error_now = error_old

        iterations_per_print = math.ceil(n_iterations/10)
        if verbose and thisIteration % max(1, iterations_per_print) == 0:
            seconds = time.time() - start_time_last_verbose
            print(
                f'SCG: Iteration {iteration:d} ObjectiveF={eval_f(error_now):.5f} Scale={beta:.3e} Seconds={seconds:.3f}')
            start_time_last_verbose = time.time()
        if save_wtrace:
            wtrace.append(w.copy())
        ftrace.append(eval_f(error_now))

        if success:

            error_old = error_new
            gradold[:] = gradnew
            gradnew[:] = error_gradient_f(w, *fargs)

            # If the gradient is zero then we are done.
            gg = gradnew.T @ gradnew
            if gg == 0:
                return {'w': w,
                        'f': error_now,
                        'n_iterations': iteration,
                        'wtrace': np.array(wtrace)[:iteration + 1, :] if save_wtrace else None,
                        'ftrace': np.array(ftrace)[:iteration + 1],
                        'reason': 'zero gradient',
                        'time': time.time() - start_time}

        if isnan(Delta) or Delta < 0.25:
            beta = min(4.0 * beta, betamax)
        elif Delta > 0.75:
            beta = max(0.5 * beta, betamin)

        # Update search direction using Polak-Ribiere formula, or re-start
        # in direction of negative gradient after nparams steps.
        if nsuccess == nvars:
            d[:] = -gradnew
            nsuccess = 0
        elif success:
            gamma = (gradold - gradnew).T @ (gradnew / mu)
            d[:] = gamma * d - gradnew

        thisIteration += 1
        iteration += 1

        # If we get here, then we haven't terminated in the given number of iterations.

        # a = {
        #     "beta": beta,
        #     "d": d,
        #     "error_now": error_now,
        #     "error_old": error_old,
        #     "ftrace": ftrace,
        #     "gradnew": gradnew,
        #     "gradold": gradold,
        #     "kappa": kappa,
        #     "mu": mu,
        #     "nsuccess": nsuccess,
        #     "nvars": nvars,
        #     "step": iteration,
        #     "success": success,
        #     "theta": theta,
        #     "w": w
        # }
        # import json
        # print(json.dumps(a, sort_keys=True, indent=2, default=str))

    return {'w': w,
            'f': error_now,
            'n_iterations': iteration,
            'wtrace': np.array(wtrace)[:iteration + 1, :] if save_wtrace else None,
            'ftrace': np.array(ftrace)[:iteration + 1],
            'reason': 'did not converge',
            'time': time.time() - start_time}


if __name__ == '__main__':

    def error(w):
        return (w - 1.5)**2

    def error_grad(w):
        return 2 * (w - 1.5)

    w = np.array([-5.5], dtype=np.float32)

    result = sgd(w, error, [], 1000, error_grad,
                 learning_rate=0.1, momentum_rate=0.5)
    print(f"sgd w is {result['w'][0]:.3f}")

    result = adam(w, error, [], 1000, error_grad, learning_rate=0.1)
    print(f"adam w is {result['w'][0]:.3f}")

    result = scg(w, error, [], 1000, error_grad)
    print(f"scg w is {result['w'][0]:.3f}",
          result['n_iterations'], result['reason'])

    print('----Rosenbrock----')
    import mlx.core as mx

    def rosenbrock(xy):
        x, y = xy
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    def rosenbrock_grad(xy):
        return np.array(mx.grad(rosenbrock)(mx.array(xy)))

    w = np.array([-2, 2], dtype=np.float32)

    steps = 1000
    result = sgd(w, rosenbrock, [], steps, rosenbrock_grad,
                 learning_rate=0.001, save_wtrace=True)
    print(f"sgd w is {result['w']}")
    path_sgd = result['wtrace']

    result = adam(w, rosenbrock, [], steps, rosenbrock_grad,
                  learning_rate=0.05, save_wtrace=True)
    print(f"adam w is {result['w']}")
    path_adam = result['wtrace']

    result = scg(w, rosenbrock, [], steps, rosenbrock_grad, save_wtrace=True)
    print(f"scg w is {result['w']}", result['n_iterations'], result['reason'])
    path_scg = result['wtrace']

    import matplotlib.pyplot as plt

    def plot_rosenbrok(paths, names, colors):
        assert len(paths) == len(names) == len(colors), ValueError
        n = 300
        x = mx.linspace(-2.5, 1.5, n)
        y = mx.linspace(-1.5, 3.5, n)
        minimum = (1.0, 1.0)

        X, Y = mx.meshgrid(x, y)
        Z = rosenbrock([X, Y])

        fig = plt.figure(figsize=(8, 5))

        ax = fig.add_subplot(1, 1, 1)
        ax.contour(X, Y, Z, levels=40, cmap='inferno')
        ax.contourf(X, Y, Z, levels=40, cmap='binary', alpha=0.7)

        for path, name, color in zip(paths, names, colors):
            iter_x, iter_y = path[:, 0], path[:, 1]
            ax.plot(iter_x, iter_y, marker='x', ms=3,
                    lw=2, label=name, color=color)
        ax.legend(fontsize=12)
        ax.axis('off')
        ax.plot(*minimum, 'kD')
        ax.set_title(
            'Rosenbrok Function: $f(x, y) = (1 - x)^2 + 100(y - x^2)^2$')
        fig.tight_layout()
        plt.show()

    freq = 1
    paths = [path_adam[::freq], path_sgd[::freq], path_scg[::freq]]
    names = ['Adam', 'SGD', 'SCG']
    colors = ['royalblue', 'seagreen', 'red']
    print('saving results to rosenbrock.png')
    plot_rosenbrok(paths, names, colors)
