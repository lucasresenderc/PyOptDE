"""
Differential Evolution class file.
"""

import json
import numpy as np

from .single_differential_evolution import SingleDifferentialEvolution
from .multiple_differential_evolution import MultipleDifferentialEvolution


class DifferentialEvolution:
    """Differential Evolution model.
    """

    def __init__(self, f, n, N=False, crossover_p=False, scaling_factor=.75, populate_method='cube', populate_data=(0, 1), iterations=100, base_change=False, get_history=False, seed=False, trials=1):

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

        self.results = {}

    def run(self, processes=1):
        if self.trials == 1:
            opt = SingleDifferentialEvolution(
                self.f,
                self.n,
                N=self.N,
                crossover_p=self.cr,
                scaling_factor=self.F,
                populate_method=self.populate_method,
                populate_data=self.populate_data,
                iterations=self.iter,
                base_change=self.base_change,
                get_history=self.get_history,
                seed=self.seed
            )

            data = opt.run()

            if self.get_history:
                results = {
                    'generations': data[0],
                    'f_generations': data[1],
                    'index_bests': data[2],
                    'general_enhances_history': data[3],
                    'best_enhances_history': data[4]
                }

                if self.base_change:
                    results['ill_conditioned'] = data[5]
            else:
                results = {
                    'x': data[0],
                    'fx': data[1],
                    'count_general_enhances': (1.*data[2])/(self.N*self.iter),
                    'count_best_enhances': (1.*data[3])/self.iter
                }

                if self.base_change:
                    results['ill_conditioned'] = data[4]

        if self.trials > 1:
            opt = MultipleDifferentialEvolution(
                self.f,
                self.n,
                N=self.N,
                crossover_p=self.cr,
                scaling_factor=self.F,
                populate_method=self.populate_method,
                populate_data=self.populate_data,
                iterations=self.iter,
                base_change=self.base_change,
                get_history=self.get_history,
                seed=self.seed,
                trials=self.trials
            )

            data = opt.run(processes=processes)

            if self.get_history:
                results = {}

                for i in range(self.trials):
                    results[i] = {
                        'generations': data[i][0],
                        'f_generations': data[i][1],
                        'index_bests': data[i][2],
                        'general_enhances_history': data[i][3],
                        'best_enhances_history': data[i][4]
                    }

                    if self.base_change:
                        results[i]['ill_conditioned'] = data[i][5]
            else:
                best_x = data[0][0]
                best_f = data[0][1]
                best_ill = data[0][4] if self.base_change else False
                best_seed = 0
                mean_general_enhances = data[0][2]
                mean_best_enhances = data[0][3]
                general_enhances_at_the_best = data[0][2]
                best_enhances_at_the_best = data[0][3]

                for i in range(1, self.trials):
                    mean_general_enhances += data[i][2]
                    mean_best_enhances += data[i][3]

                    if data[i][1] < best_f:
                        best_x = data[i][0]
                        best_f = data[i][1]
                        best_ill = data[i][4] if self.base_change else False
                        best_seed = i
                        general_enhances_at_the_best = data[i][2]
                        best_enhances_at_the_best = data[i][3]

                results = {
                    'x': best_x,
                    'fx': best_f,
                    'mean_general_enhances': (1.*mean_general_enhances)/(self.trials*self.N*self.iter),
                    'mean_best_enhances': (1.*mean_best_enhances)/(self.trials*self.iter),
                    'general_enhances_at_the_best': (1.*general_enhances_at_the_best)/(self.N*self.iter),
                    'best_enhances_at_the_best': (1.*best_enhances_at_the_best)/self.iter
                }

                if self.seed:
                    results['seed_of_the_best'] = self.seed[best_seed]

                if self.base_change:
                    results['ill_conditioned_at_the_best'] = best_ill

        self.results = results

        return results

    def get_results(self):
        return self.results

    def write_results(self, filename):
        d = self.results.copy()

        if self.get_history:
            if self.trials > 1:
                for i in range(self.trials):
                    d[i]['generations'] = [g.tolist() for g in d[i]['generations']]
                    d[i]['f_generations'] = [fg.tolist() for fg in d[i]['f_generations']]
            else:
                d['generations'] = [g.tolist() for g in d['generations']]
                d['f_generations'] = [fg.tolist() for fg in d['f_generations']]
        else:
            d['x'] = d['x'].tolist()

        with open(filename, 'w') as f:
            json.dump(d, f, indent=2)

    def load_results(self, filename):
        with open(filename) as f:
            d = json.load(f)

            if self.get_history:
                if self.trials > 1:
                    for i in range(self.trials):
                        d[i] = d.pop(str(i))
                        d[i]['generations'] = [np.array(g) for g in d[i]['generations']]
                        d[i]['f_generations'] = [np.array(fg) for fg in d[i]['f_generations']]
                else:
                    d['generations'] = [np.array(g) for g in d['generations']]
                    d['f_generations'] = [np.array(fg) for fg in d['f_generations']]
            else:
                d['x'] = np.array(d['x'])

            self.results = d