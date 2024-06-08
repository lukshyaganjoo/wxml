import torch
import numpy as np
from itertools import product, combinations

# Generate binary vectors
def vectors(n, binary=False):
    if binary:
        return torch.tensor(list(product([0, 1], repeat=n)), dtype=torch.float32).T
    return torch.tensor(list(product([-1, 1], repeat=n)), dtype=torch.float32).T

# Beta function
def beta(A, binary_vectors):
    B = A @ binary_vectors
    return torch.mean(torch.max(torch.abs(B), dim=0).values)

# Currently I have rows that appear twice up to a sign change, so this funciton deletes those.
def fix_row_signs(tensor):
    # Iterate over each row
    normalized_tensor = []
    for row in tensor:
        # Find the first non-zero element for sign determination
        first_non_zero_index = (row != 0).nonzero(as_tuple=True)[0][0]
        sign = torch.sign(row[first_non_zero_index])
        normalized_row = row * sign
        normalized_tensor.append(normalized_row)
    normalized_tensor = torch.stack(normalized_tensor)
    # Now return the unique ones.
    return torch.unique(normalized_tensor, dim=0)

# Generates all possible rows appearing in the optimal matrix. Now it's a matter of checking all row combinations.
def candidate_rows(n):
  # Each row represents a subset
  binary_choices = vectors(2**n, binary=True).T

  # Sums over chosen subset to yield all candidate, unnormalized rows 
  rows = torch.Tensor(binary_choices @ vectors(n).T).float()

  # Compute their norms
  row_norms = torch.norm(rows, dim=1, keepdim=True)

  # Normalize rows
  normalized_rows_with_nans = rows/row_norms

  # During normalization, some choices add up to 0, so after normalizing we obtain all nan values
  # We thus slice the tensor to remove nan rows
  normalized_rows = normalized_rows_with_nans[~torch.all(normalized_rows_with_nans.isnan(), dim = 1)]

  # Remove redundant rows (no need to try them twice!)
  unique_normalized_rows = torch.unique(normalized_rows, dim=0)

  # We can also remove rows that are equal up to a row-wise sign change.
  final_rows = fix_row_signs(unique_normalized_rows)

  return final_rows

# Using a generator function because storing 680C4 rows is impossible (~8 billion rows)
def generate_combinations(elements, r):
    for combination in combinations(elements, r):
        yield combination