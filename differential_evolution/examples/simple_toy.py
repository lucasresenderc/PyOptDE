import numpy as np
from ..algorithm import DifferentialEvolution as DE


'''
Simplest possible use
'''


# define a function from R^n to R to minimize
def f(x):
    return np.dot(x, x)


def simple_toy():
    # set the dimension of the search space
    n = 2
    # init an DifferentialEvolution instance
    optimizer = DE(f, n, iterations=1000)
    # runs the optimization
    optimizer.run()
    # then you can save your results
    optimizer.write_results('results.json')
    # or just show then on the screen
    print(optimizer.get_results())


if __name__ == "__main__":
    simple_toy()