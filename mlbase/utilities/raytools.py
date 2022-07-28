import ray
import subprocess
import os
import time


ray_hosts = []
locale_prep = ['export', 'LC_ALL=C.utf8', ';', 'export', 'LANG=C.utf8', ';']


def ssh_on(host, command):
    if host:
        subprocess.call(['ssh', host, *locale_prep, *
                        command.split(), '>', '/dev/null', '2>&1'])
    else:
        subprocess.call(command.split())


def ray_start(primary, port, hosts=None, num_gpus=None, pythonpath=None):
    """
    Params:
    -------
    primary: host to start on (must be the same as invoking machine)
    port: port to start host on
    hosts: list of machine host names to spawn slaves on ([mars, denver])
    num_gpus: gpus to use for each ray process
    pythonpath: path(s) pointing to where some python modeules reside for worker node
                ('/s/chopin/l/grad/stock/research/mlbase')
    """
    global ray_hosts

    if hosts:
        print(f'Starting ray locally on [{primary}] and on hosts {hosts}')
    else:
        print(f'Starting ray locally on {primary}.')

    host = os.uname()[1]
    if host != primary:
        raise Exception(f'ray_start must be run on {primary}')

    # additional ssh/ray command settings
    gpu_str = f' --num-gpus={num_gpus} ' if num_gpus and num_gpus > 0 else ''
    pypath = ''
    if pythonpath:
        # TODO: add check to append ${PYTHONPATH}:' if len(path) > 0
        os.environ['PYTHONPATH'] = pythonpath
        pypath = f'export PYTHONPATH={pythonpath} ;'

    path = os.getcwd()
    ssh_on(
        None, f'ray --logging-level=error start --head --port={port}' + gpu_str)
    if hosts:
        for host in hosts:
            ssh_on(
                host, f'cd {path} ; hostname ; {pypath} ray --logging-level=error start --address={primary}.cs.colostate.edu:{port}' + gpu_str)
    ray_hosts = hosts
    time.sleep(5)

    print('-------------------calling ray.init-------------------')
    ray.init(address=f'{primary}.cs.colostate.edu:{port}',
             log_to_driver=True, ignore_reinit_error=True)


def ray_stop():
    global ray_hosts

    ray.shutdown()
    time.sleep(1)
    if ray_hosts:
        for host in ray_hosts:
            ssh_on(host, 'ray stop')
    ssh_on(None, 'ray stop')


if __name__ == '__main__':
    primary = os.uname()[1]
    port = 6813
    ray_start(primary, port, ['mars', 'denver'],
              pythonpath='/s/chopin/l/grad/stock/research/mlbase')

    @ray.remote
    def f(i):
        time.sleep(5)
        try:
            p = os.environ['PYTHONPATH']
        except KeyError:
            p = None
        print(os.uname()[1], p, i)
        return os.uname()[1], i

    start_t = time.time()
    o = ray.get([f.remote(i) for i in range(30)])
    print(o)

    print(
        f'Finished in: {(time.time() - start_t):0.4f} seconds')

    ray_stop()
