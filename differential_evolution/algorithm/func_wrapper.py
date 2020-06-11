from .single_differential_evolution import SingleDifferentialEvolution

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
