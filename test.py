from evotorch import Problem
from evotorch.algorithms import SNES
import torch

g = 0
def sphere(x: torch.Tensor) -> torch.Tensor:
    global g
    g += 1
    result = torch.sum(x.pow(2.0))
    print(result, result.dim())
    return result


problem = Problem("min", sphere, solution_length=10, initial_bounds=(-1, 1))
searcher = SNES(problem, popsize=20, stdev_init=5)
searcher.run(100)

print(g)