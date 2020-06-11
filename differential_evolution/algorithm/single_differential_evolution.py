"""
Single Differential Evolution class file.
"""

import numpy as np
import scipy.linalg as la

from functools import partial
from numpy.random import default_rng
from scipy.optimize import minimize

from .utils import p_cr


class SingleDifferentialEvolution:
    """Single Differential Evolution model.
    """

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
                try:
                    acT = auxacT
                    acQ = la.inv(auxacQ)
                except:
                    ill_conditioned = gen

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
            self.cr = minimize(partial(p_cr, self.n, self.F), x0=np.array([.5]), method='Nelder-Mead', options={'xatol': 1e-10}).x[0]

        if self.base_change:
            return self.normalized_de()
        else:
            return self.usual_de()