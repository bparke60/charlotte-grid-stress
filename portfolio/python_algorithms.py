"""Original implementations of introductory Python algorithm exercises."""
from collections import Counter

def center_list(values):
    if not values:
        raise ValueError("Cannot center an empty list")
    mean = sum(values) / len(values)
    return mean, [value - mean for value in values]

def matrix_replacement(matrix):
    return [[0 if item is None else item for item in row] for row in matrix]

def histogram(values):
    return dict(Counter(values))

def flatten(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
