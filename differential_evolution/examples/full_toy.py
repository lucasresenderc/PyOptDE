import numpy as np
from ..algorithm import DifferentialEvolution as DE


'''
TEST FUNCTIONS
'''


def f1(x):
    return np.dot(x, x)


def f2(x):
    return np.sum(x*x-10*np.cos(2*np.pi*x)) + 10*x.size


def f3(x):
    return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)


def f4(x):
    aux = x*x
    return np.sum(.5*aux*aux - 8*aux + 2.5*x) + 40*x.size


def f5(x):
    aux1 = np.sum(x*x)/4000
    aux2 = np.prod(np.cos(x/np.sqrt(np.arange(1, x.size+1))))
    return aux1 - aux2 + 1


def full_toy():
    '''
    PARAMETERS TO TEST
    '''
    functions = [
        f1,
        f2,
        f3,
        f4,
        f5
    ]

    intervals = [
        (-1, 1),
        (-5.12, 5.12),
        (-5.12, 5.12),
        (-5, 5),
        (-600, 600)
    ]

    F = 0.75
    iterations = 50005
    trials = 100
    for n in [5, 50]:
        for cr in [False, .5]:
            for base_change in [False, 100]:
                for i in range(1):
                    filename = 'results/f{}-n={}-cr={}-base_change={}.json'.format(i+1, n, cr, base_change)
                    print('running {}'.format(filename))
                    optimizer = DE(
                        functions[i],
                        n,
                        N=2*n,
                        crossover_p=cr,
                        scaling_factor=F,
                        populate_method='cube',
                        populate_data=intervals[i],
                        iterations=iterations,
                        base_change=base_change,
                        get_history=True,
                        seed=range(1, trials+1),
                        trials=trials
                    )
                    optimizer.run(processes=trials)
                    optimizer.write_results(filename)


if __name__ == "__main__":
    full_toy()