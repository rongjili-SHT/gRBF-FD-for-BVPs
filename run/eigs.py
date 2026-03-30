import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import multiprocessing as mp
import time
import numpy as np
from problems.eigs_semi_torus import eigs_semi_torus
from problems.eigs_semi_sphere import eigs_semi_sphere

def run_experiment(args):
    i, N, j, seed, num_eigs = args
    try:
        evals, evecs = eigs_semi_torus(N=N, l=3, K=20, l_grad=3, K_grad=25, num_eigs=num_eigs, seed=seed)
        print(f"[Done] N={N}, seed={seed}", flush=True)
        return (i, j, evals)
    except Exception as e:
        print(f"[Error] N={N}, seed={seed}, failed: {e}", flush=True)
        return (i, j, np.nan)

if __name__ == '__main__':
    N_vals = [1600, 3200, 6400, 12800, 25600, 51200]
    seeds = np.arange(12)

    num_eigs = 20

    results = np.zeros((len(N_vals), len(seeds), num_eigs))

    tasks = []
    for i, N in enumerate(N_vals):
        for j, seed in enumerate(seeds):
            tasks.append((i, N, j, seed, num_eigs))

    tasks.sort(key=lambda x: x[1], reverse=True)

    num_cores = min(mp.cpu_count(), 60) 
    
    start_time = time.time()

    with mp.Pool(processes=num_cores) as pool:
        for res in pool.imap_unordered(run_experiment, tasks):
            i, j, evals = res
            results[i, j, :] = evals

    end_time = time.time()
    print(f"All computations finished in {(end_time - start_time)/60:.2f} minutes.")

    np.save('./data/eigs_semi_torus_d3.npy', results)