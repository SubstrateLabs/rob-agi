from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_695367ec.main import solve_695367ec

def print_grid(grid):
    for row in grid.values:
        print(' '.join(str(cell) for cell in row))
    print()

# Test cases
test_cases = [
    ColoredGrid(values=[[1]]),  # 1x1
    ColoredGrid(values=[[2, 2], [2, 2]]),  # 2x2
    ColoredGrid(values=[[3, 3, 3], [3, 3, 3], [3, 3, 3]]),  # 3x3
    ColoredGrid(values=[[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4]]),  # 4x4
    ColoredGrid(values=[[5, 5, 5, 5, 5], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5]])  # 5x5
]

for i, test_case in enumerate(test_cases):
    print(f"Test case {i + 1} ({test_case.get_dimensions()[0]}x{test_case.get_dimensions()[1]}):")
    result = solve_695367ec(test_case)
    print_grid(result)
