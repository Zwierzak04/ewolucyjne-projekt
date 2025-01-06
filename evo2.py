from evotorch import Problem
from evotorch.algorithms import SNES, CMAES
from evotorch.logging import StdOutLogger
import torch
import numpy as np
from shapely import LineString
import itertools
import functools
import math


# FIXME: czasami z jakiegoś powodu nie działa
def liczba_przeciec(edges, positions: torch.Tensor) -> torch.Tensor:
    reshaped = positions.view(-1, 2)
    edge_pos = [LineString(reshaped[e]) for e in edges]
    intersectins = 0.0
    for i1, i2 in itertools.combinations(range(len(edges)), 2):
        intersection = edge_pos[i1].intersection(edge_pos[i2])

        # brak przecięcia
        if intersection.is_empty:
            continue

        # linie na siebie nachodzą - nieskończoność przecięć
        if(intersection.geom_type != 'Point'):
            return torch.tensor(math.inf)

        # pomijamy krawędzie, które mają wspólne wierzchołki
        # jeśli na siebie nie nachodzą to nie ma sensu ich sprawdzać
        if set(edges[i1]) & set(edges[i2]):
            continue

        intersectins += 1

    return torch.tensor(intersectins)

def liczba_przeciec_batched(edges, positions_batch: torch.Tensor) -> torch.Tensor:
    """
    Calculates the number of intersections for a batch of position tensors.
    
    Args:
        edges: List of edges defined by vertex indices.
        positions_batch: A tensor of shape (batch_size, num_vertices, 2).
    
    Returns:
        A tensor of shape (batch_size,) where each element is the number of intersections for a respective batch.
    """
    batch_size = positions_batch.size(0)
    results = torch.zeros(batch_size)

    for batch_idx in range(batch_size):
        positions = positions_batch[batch_idx]
        reshaped = positions.view(-1, 2)
        edge_pos = [LineString(reshaped[e]) for e in edges]
        intersections = 0.0

        for i1, i2 in itertools.combinations(range(len(edges)), 2):
            intersection = edge_pos[i1].intersection(edge_pos[i2])

            # No intersection
            if intersection.is_empty:
                continue

            # Infinite intersections (lines overlap)
            if intersection.geom_type != 'Point':
                results[batch_idx] = float('inf')
                break

            # Ignore edges that share vertices unless they overlap
            if set(edges[i1]) & set(edges[i2]):
                continue

            intersections += 1

        if results[batch_idx] != float('inf'):
            results[batch_idx] = intersections

    return results

import torch

def what(edges, positions_batch: torch.Tensor) -> torch.Tensor:
    """
    Fully vectorized calculation of intersections for a batch of position tensors.

    Args:
        edges: List of edges defined by vertex indices.
        positions_batch: A tensor of shape (batch_size, num_vertices, 2).

    Returns:
        A tensor of shape (batch_size,) with the number of intersections for each batch.
    """
    batch_size, num_vertices, _ = positions_batch.shape
    num_edges = len(edges)
    
    # Step 1: Extract edge start and end points
    edge_indices = torch.tensor(edges, dtype=torch.long)
    p1 = positions_batch[:, edge_indices[:, 0], :]  # (batch_size, num_edges, 2)
    p2 = positions_batch[:, edge_indices[:, 1], :]  # (batch_size, num_edges, 2)
    
    # Step 2: Pairwise combinations of edges
    edge_combinations = torch.combinations(torch.arange(num_edges), 2, with_replacement=False)
    e1_idx, e2_idx = edge_combinations[:, 0], edge_combinations[:, 1]
    
    p1_e1, p2_e1 = p1[:, e1_idx], p2[:, e1_idx]  # (batch_size, num_combinations, 2)
    p1_e2, p2_e2 = p1[:, e2_idx], p2[:, e2_idx]  # (batch_size, num_combinations, 2)
    
    # Step 3: Vectorized intersection calculation
    d1 = p2_e1 - p1_e1  # Directions of edge 1
    d2 = p2_e2 - p1_e2  # Directions of edge 2
    r = p1_e1 - p1_e2   # Vector between start points
    
    det = d1[:, :, 0] * d2[:, :, 1] - d1[:, :, 1] * d2[:, :, 0]  # Determinant (batch_size, num_combinations)
    det_nonzero = det != 0  # Filter out parallel lines
    
    t = (r[:, :, 0] * d2[:, :, 1] - r[:, :, 1] * d2[:, :, 0]) / det  # Parametric t (batch_size, num_combinations)
    u = (r[:, :, 0] * d1[:, :, 1] - r[:, :, 1] * d1[:, :, 0]) / det  # Parametric u (batch_size, num_combinations)
    
    valid_t = (t >= 0) & (t <= 1)
    valid_u = (u >= 0) & (u <= 1)
    valid_intersections = det_nonzero & valid_t & valid_u  # Valid intersections (batch_size, num_combinations)
    
    # Step 4: Ignore edges sharing vertices
    shared_vertices = (
        (edge_indices[e1_idx, None] == edge_indices[e2_idx, None, :]).any(dim=-1)
    )  # Shape: (num_combinations,)
    shared_vertices = shared_vertices.unsqueeze(0).expand(batch_size, -1)
    valid_intersections &= ~shared_vertices
    
    # Step 5: Count intersections
    intersections_count = valid_intersections.sum(dim=1)  # (batch_size,)
    
    return intersections_count


def search_best(n_verts, edges):
    # zabije sie
    problem = Problem(
        'min',
        functools.partial(liczba_przeciec, edges),
        solution_length=n_verts*2,
        #vectorized=True,
        initial_bounds=(-1, 1),
    )
    snes = CMAES(problem, popsize=100, stdev_init=5)
    logger = StdOutLogger(searcher=snes, interval=10)
    snes.run(100)

    return snes.status['best']
