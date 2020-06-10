import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.special import comb
from scipy.special import betainc, beta
from scipy.optimize import minimize
from functools import partial
from numpy.random import default_rng
from multiprocessing import Pool
import json
import pandas as pd


def p(k, F):
    g = 2 * F * F + 1
    a = g / (1 + g)
    return np.power(g, -.5 * k) * np.power(a, k) / (k * beta(.5 * k, .5 * k + 1)) + 1 - betainc(.5 * k, .5 * k + 1, a)


def b(n, k, CR):
    return comb(n, k, exact=True) * np.power(CR, k) * np.power(1 - CR, n - k)


def pcr(n, F, CR):
    aux = 0
    CR = CR[0]
    for k in range(1, n + 1):
        aux += p(k, F) * b(n, k, CR)
    return -aux


class SingleDifferentialEvolution:
    def __init__(self, f, n, N=False, crossover_p=False, scaling_factor=.75, populate_method='cube', populate_data=(0, 1), iterations=100, base_change=False, get_history=False, seed=False):
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

        self.rng = default_rng()
        if seed:
            self.rng = default_rng(seed)

    def populate(self):
        # generates population
        if self.populate_method == 'given':
            pop = self.populate_data
        elif self.populate_method == 'sphere':
            loc = self.populate_data[0]
            radius = self.populate_data[1]
            pop = np.zeros(shape=(self.n, self.N))
            for i in range(self.N):
                ind = self.rng.normal(size=self.n)
                ind /= np.linalg.norm(ind)
                pop[:, i] = loc + ind * np.power(self.rng.random(), 1 / self.n)*radius
        else:
            low = self.populate_data[0]
            high = self.populate_data[1]
            pop = low + (high - low) * self.rng.random((self.n, self.N))

        # compute f values of pop
        fg = np.zeros(shape=self.N)
        for i in range(self.N):
            fg[i] = self.f(pop[:, i])

        return pop, np.array(fg)

    def usual_de(self):
        # initializes the population
        g, fg = self.populate()

        count_general_enhances = 0
        count_best_enhances = 0

        if self.get_history:
            generations = [g]
            f_generations = [fg]
            i_bests = [np.argmin(fg).tolist()]
            general_enhances_history = []
            best_enhances_history = []

        for gen in range(self.iter - 1):
            if self.get_history:
                best_enhances_history.append(0)
                general_enhances_history.append(0)

            # iterate
            new_g = np.zeros(shape=(self.n, self.N))
            new_fg = np.zeros(shape=self.N)
            for i in range(self.N):
                # choose tree
                b, c, d = np.union1d(np.arange(i), np.arange(i + 1, self.N))[self.rng.choice(self.N - 1, 3, replace=False)]
                # compute the difference to sum
                dif = g[:, b] + self.F * (g[:, c] - g[:, d])
                # filter with k
                k = 1
                if self.n > 1:
                    k = self.rng.binomial(self.n, self.cr)
                k_filter = np.zeros(self.n)
                k_filter[self.rng.choice(self.n, k, replace=False)] = np.ones(k)
                k_filter[self.rng.integers(self.n)] = 1
                dif = dif * k_filter

                fo = self.f(g[:, i] * (np.ones(self.n) - k_filter) + dif)
                # verify if change or not
                if fo < fg[i]:
                    new_g[:, i] = g[:, i] * (np.ones(self.n) - k_filter) + dif
                    new_fg[i] = fo
                    count_general_enhances += 1
                    if self.get_history:
                        general_enhances_history[-1] += 1
                else:
                    new_g[:, i] = g[:, i]
                    new_fg[i] = fg[i]

            # verify if the best improves
            if np.min(new_fg) < np.min(fg):
                count_best_enhances += 1
                if self.get_history:
                    best_enhances_history[-1] += 1

            g = new_g
            fg = new_fg

            if self.get_history:
                generations.append(g)
                f_generations.append(fg)
                i_bests.append(np.argmin(fg).tolist())

        if self.get_history:
            return generations, f_generations, i_bests, general_enhances_history, best_enhances_history
        else:
            i_best = np.argmin(fg)
            return g[:, i_best], fg[i_best], count_general_enhances, count_best_enhances

    def compute_trafo(self, X):
        # get mean
        T = np.array([1 / self.N * np.sum(X, axis=1)]).T * np.array([np.ones(self.N)])
        # M is positive semidef
        M = np.dot((X - T), (X - T).T)
        # root
        S = la.sqrtm(M)
        # root
        Q = la.inv(S)
        # returns
        new_X = np.dot(Q, X - T)
        return new_X, Q, T

    def normalized_de(self):
        # initializes the population
        g, fg = self.populate()
        norm_g = np.copy(g)

        count_general_enhances = 0
        count_best_enhances = 0

        if self.get_history:
            generations = [g]
            f_generations = [fg]
            i_bests = [np.argmin(fg).tolist()]
            general_enhances_history = []
            best_enhances_history = []

        # initializes the transformation
        acT = np.zeros(shape=(self.n, self.N))
        acQ = np.identity(self.n)
        invacQ = np.identity(self.n)
        ill_conditioned = False

        for gen in range(self.iter - 1):
            if self.get_history:
                best_enhances_history.append(0)
                general_enhances_history.append(0)

            # change basis with frequence base_change
            if gen % self.base_change == 0 and gen > 0 and not ill_conditioned:
                #verify if the condition number ins't too high
                auxg, Q, T = self.compute_trafo(g)
                auxacT = acT + np.dot(la.inv(acQ), T)
                auxacQ = np.dot(Q, acQ)
                cond = np.linalg.cond(auxacQ)
                if cond < 1e8:
                    acT = auxacT
                    acQ = la.inv(auxacQ)
                else:
                    ill_conditioned = gen
                    print(ill_conditioned)

            # iterate
            new_g = np.zeros(shape=(self.n, self.N))
            norm_new_g = np.zeros(shape=(self.n, self.N))
            new_fg = np.zeros(shape=self.N)

            for i in range(self.N):
                # choose tree
                b, c, d = np.union1d(np.arange(i), np.arange(i + 1, self.N))[self.rng.choice(self.N - 1, 3, replace=False)]
                # compute the difference to sum
                dif = g[:, b] + self.F * (g[:, c] - g[:, d])
                # filter with k
                k = 1
                if self.n > 1:
                    k = self.rng.binomial(self.n, self.cr)
                k_filter = np.zeros(self.n)
                k_filter[self.rng.choice(self.n, k, replace=False)] = np.ones(k)
                k_filter[self.rng.integers(self.n)] = 1
                dif = dif * k_filter
                # verify if change or not
                o = g[:, i] * (np.ones(self.n) - k_filter) + dif
                norm_o = np.dot(invacQ, o) + acT[:, 0]
                fo = self.f(norm_o)
                if fo < fg[i]:
                    new_g[:, i] = o
                    norm_new_g[:, i] = norm_o
                    new_fg[i] = fo
                    count_general_enhances += 1
                    if self.get_history:
                        general_enhances_history[-1] += 1
                else:
                    new_g[:, i] = g[:, i]
                    norm_new_g[:, i] = norm_g[:, i]
                    new_fg[i] = fg[i]

            # verify if the best improves
            if np.min(new_fg) < np.min(fg):
                count_best_enhances += 1
                if self.get_history:
                    best_enhances_history[-1] += 1

            g = new_g
            norm_g = norm_new_g
            fg = new_fg

            if self.get_history:
                generations.append(g)
                f_generations.append(fg)
                i_bests.append(np.argmin(fg).tolist())

        if self.get_history:
            return generations, f_generations, i_bests, general_enhances_history, best_enhances_history, ill_conditioned
        else:
            i_best = np.argmin(fg)
            return g[:, i_best], fg[i_best], count_general_enhances, count_best_enhances, ill_conditioned

    def run(self):
        # get optimum cr for execution if no cr was given
        if not self.cr:
            self.cr = minimize(partial(pcr, self.n, self.F), x0=np.array([.5]), method='Nelder-Mead', options={'xatol': 1e-10}).x[0]

        if self.base_change:
            return self.normalized_de()
        else:
            return self.usual_de()


def single_realization(params):
    f, n, N, cr, F, populate_method, populate_data, iterations, base_change, get_history, seed = params
    opt = SingleDifferentialEvolution(
        f,
        n,
        N=N,
        crossover_p=cr,
        scaling_factor=F,
        populate_method=populate_method,
        populate_data=populate_data,
        iterations=iterations,
        base_change=base_change,
        get_history=get_history,
        seed=seed
    )
    return opt.run()


class MultipleDifferentialEvolution:
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


class DifferentialEvolution:
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
