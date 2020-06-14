![DELOGO](https://user-images.githubusercontent.com/24375125/84321286-3f3ac580-ab49-11ea-8d0d-b841bc06113c.png)
# DifferentialEvolution

This module is an implementation of the Differential Evolution (DE) algorithm. Proposed by Price and Storn in a series of papers [1, 2, 3],  the Differential Evolution is a along-established evolutionary algorithm that aims to optimize functions on a continuous domain.

The Differential Evolution requires no qualitative nor quantitative information about the objective function (such as continuity, differentiability, etc). The algorithm is useful and recommended to deal with black-boxes or functions with many local optima that might trick other algorithms.

The idea behind the algorithm is to construct and improve populations (sets of points on the domain) iteration after iteration in a way that those points will reach the optima. 

This implementation assumes that the domain is R^n and that the objective function is real. There are several versions of the DE algorithm, this implementation uses the DE/rand/1/bin (see 4). The following code illustrates the basic usage (also see toy.py):


```python
import numpy as np
from DE import DifferentialEvolution as DE

'''
Simplest possible use
'''
# define a function from R^n to R to minimize
def f(x):
  return np.dot(x, x)

# set the dimension of the search space
n = 2
# init an DifferentialEvolution instance
optimizer = DE(f, n, iterations=1000, seed=1)
# runs the optimization
optimizer.run()
# then you can save your results
optimizer.write_results('results.json')
# or just show then on the screen
print(optimizer.get_results())
```

Returns at results.json:
```json
{
  "x": [
    -0.00018860256872277426,
    0.10069513518651194
  ],
  "fx": 0.010139545821158842,
  "count_general_enhances": 0.1735,
  "count_best_enhances": 0.242
}
```

Where:
- ***x:*** is the optima;
- ***fx:*** is the value of f(x);
- ***count_general_enhances:*** returns the mean number of enhancements per individual per generation. Its a good metric to know if the algorithm is improving the solution or not. If this value is low you should try change the crossover probability or the scaling factor;
- ***count_best_enhances:*** returns the mean number of enhancements of the best individual per generation. Its a good metric to know if the algorithm is improving the solution or not. If this value is low you should try change the crossover probability or the scaling factor.

___
## Parallelizing

Since it is an stochastic algorithm one might run it at least a few times to make sure of the results. To run the algorithm several times in parallel use:

```python
import numpy as np
from DE import DifferentialEvolution as DE

# define a function from R^n to R to minimize
def f(x):
  return np.dot(x, x)

# set the dimension of the search space
n = 2
# init a DifferentialEvolution instance with 10 trials
optimizer = DE(f, n, iterations=1000, trials=10)
# runs the optimization using 5 processors
optimizer.run(processes=5)
# then you can save your results
optimizer.write_results('results.json')
```

It is also possible to entry with one seed to each trial:

```python
optimizer = DE(f, n, iterations=1000, trials=10, seed=range(1,11))
```
Returning:
```json
{
  "x": [
    -1.633857962321789e-18,
    4.967348489700612e-19
  ],
  "fx": 2.916237351223618e-36,
  "mean_general_enhances": 0.2606,
  "mean_best_enhances": 0.3434,
  "general_enhances_at_the_best": 0.28925,
  "best_enhances_at_the_best": 0.382,
  "seed_of_the_best": 6
}
```
Where:
- ***x:*** is the optima;
- ***fx:*** is the value of f(x);
- ***mean_general_enhances:*** returns the mean number of enhancements per individual per generation over all trials;
- ***mean_best_enhances:*** returns the mean number of enhancements of the best individual per generation over all trials;
- ***general_enhances_at_the_best:*** returns the mean number of enhancements per individual per generation on the best trial;
- ***best_enhances_at_the_best:*** returns the mean number of enhancements of the best individual per generation on the best trial;
- ***seed_of_the_best (only if seed is provided):*** seed from which the optima over all trials were reached.

___
## An overview of the parameters

The following code shows the full use of the algorithm (also see full.py):

```python
optimizer = DE(
                    f,
                    n,
                    N=2*n,
                    crossover_p=False,
                    scaling_factor=0.75,
                    populate_method='cube',
                    populate_data=(0,1),
                    iterations=100,
                    base_change=False,
                    get_history=False,
                    seed=False,
                    trials=1
                )
optimizer.run(processes=1)
```

The parameters are:
- ***f:*** the objective function, must take a numpy array of size ***n*** as input and returns a real number;
- ***n:*** the dimension of the search space;
- ***N (optional, default is 2n or 4 if n<2):*** population size (see [4]);
- ***crossover_p (optional, default is the optimum value from [5]):*** crossover probability (see [4]);
- ***scaling_factor (optional, default is 0.75):*** scaling factor (see [4]);
- ***populate_method (optional, default is cube):*** method to sort the initial population, must be 'cube', 'sphere' or 'given':
  - If 'cube' then the initial population will be drawn independently and uniformly on the cube [low, high]^n where ***populate_data=(low,hign)***;
  - If 'sphere' then the initial population will be drawn independently and uniformly on the sphere with center ***loc*** and radius ***radius***  where ***populate_data=(loc,radius)***;
  - If 'given' then the initial population will be given by the user at the parameter ***populate_data***, it must be a ndarray of shape (n,N) where each individual is a column;
- ***populate_data (optional, default is (0,1)):*** as explained above;
- ***iterations (recommended, default is 100)***: the number of iterations until stop;
- ***base_changes (optional, default is False)***: False if the user don't want to change basis during the algorithm or an integer ***k*** such that after each ***k*** iterations the algorithm will perform the change of basis (see [5]);
- ***get_history (optional, default is False)***: True if the user wants to output the full history of the iterations, returning the stage of the algorithm at each iteration;
- ***seed (optional, default is False)***: an integer if ***trials == 1*** or an list of ***k > 1*** integers if ***trials == k***;
- ***trials (optional, default is 1)***: the number of trials, each one wil execute the algorithm again with a different seed.

And also for the ***run()*** method:
- ***processes (optional, default is 1)***: the number of parallel processes to use.

___
## Reference

[1] Rainer Storn. *On the usage of differential evolution for function optimization.* In Proceedings of North American Fuzzy Information Processing, pages 519–523. IEEE, 1996.

[2] Rainer Storn and Kenneth Price. *Differential evolution–a simple and efficient heuristic for global optimization over continuous spaces.* Journal of global optimization, 11(4):341–359, 1997.

[3] Kenneth V Price. *Differential evolution: a fast and simple numerical optimizer.* In Proceedings of North American Fuzzy Information Processing, pages 524–527. IEEE, 1996.

[4] Karol R Opara and Jaroslaw Arabas. *Differential evolution: A survey of theoretical analyses.* Swarm and evolutionary computation, 44:546–558, 2019.

[5] (wait)
