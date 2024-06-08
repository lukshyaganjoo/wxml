import torch
from itertools import combinations
from utils import *
import os

# Example list of 680 elements
elements = range(25)

n = 3
beta_max = 0
final_rows = candidate_rows(n)
elements = range(final_rows.shape[0])

binary_vectors = vectors(n)

combinations_generator = generate_combinations(elements, n)

# Define a generator function to yield combinations
def generate_combinations(elements, r):
    for combination in combinations(elements, r):
        yield combination

# Define a function to process each combination on GPU
def process_combination_on_gpu(combination):
    # Convert the combination to a tensor and move it to GPU
    combination_tensor = torch.tensor(combination, dtype=torch.float32, device='cuda')
    # Example processing (replace with actual logic)
    result = torch.sum(combination_tensor).item()
    return result

import multiprocessing

# Define your function to be applied to each element
def beta_from_combination(combination):
    # Apply your function here
    A = final_rows[list(combination)]
    return beta(A, binary_vectors)

# Define a function to compute the maximum value in parallel
def compute_max_parallel(generator, func, num_processes=None):
    pool = multiprocessing.Pool(num_processes)
    results = pool.imap_unordered(func, generator)
    pool.close()
    pool.join()
    return max(results)

# Usage example
if __name__ == "__main__":
    # Example of how to use it
    combinations_generator = generate_combinations(elements, n)
    max_value = compute_max_parallel(combinations_generator, beta_from_combination, 1000)
    print("Maximum value:", max_value)