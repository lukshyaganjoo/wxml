import torch
from itertools import combinations, product
from utils import *
from tqdm import tqdm
import timeit
import scipy.special as special # For binomial coefficient computation 

if __name__ == "__main__":

    n = 4

    final_rows = candidate_rows(n)

    # print(final_rows)

    elements = torch.tensor(range(final_rows.shape[0]))

    combinations_generator = generate_combinations(elements, n)

    binary_vectors = vectors(n)

    start = timeit.timeit()

    beta_max = 0

    best_A = None

    total = special.binom(final_rows.shape[0], final_rows.shape[1]).astype(int)

    print(total)
    count = 0
    for combination in tqdm(combinations_generator, total = total):
        
        A = final_rows[torch.tensor(combination)]
        
        beta_candidate = beta(A, binary_vectors)

        if beta_candidate > beta_max:  beta_max, best_A = beta_candidate, A

    print(f"Beta = {beta_max}")
    
    end = timeit.timeit()
    print(f"Sequential took {start-end} seconds!")