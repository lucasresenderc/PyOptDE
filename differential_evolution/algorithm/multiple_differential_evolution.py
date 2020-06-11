"""
Mulitple Differential Evolution class file.
"""

from multiprocessing import Pool

from .func_wrapper import single_realization


class MultipleDifferentialEvolution:
    """Multiple Differential Evolution model.
    """
    
    def __init__(self, f, n, N=False, crossover_p=False, scaling_factor=.75, populate_method='cube', populate_data=(0, 1), iterations=100, base_change=False, get_history=False, seed=False, trials=2):

        self.f = f
        self.n = n
        self.N = N

        if not N:
            self.N = 2*self.n if n > 2 else 4

        self.cr = crossover_p
        self.F = scaling_factor
        self.populate_method = populate_method
        self.populate_data = populate_data

        self.iter = iterations
        self.base_change = base_change

        self.get_history = get_history

        self.trials = trials
        self.seed = seed

    def run(self, processes=1):
        # generates params
        pool_params = []

        for i in range(self.trials):
            params = [
                self.f,
                self.n,
                self.N,
                self.cr,
                self.F,
                self.populate_method,
                self.populate_data,
                self.iter,
                self.base_change,
                self.get_history
            ]

            if not self.seed:
                params.append(False)
            else:
                params.append(self.seed[i])

            pool_params.append(params)

        #runs the pool
        pool = Pool(processes)

        return pool.map(single_realization, pool_params)