from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_ac3e2b04.main import solve_ac3e2b04

def print_grid(grid):
    for row in grid.values:
        print(' '.join(str(cell) for cell in row))

# Test case from example_3
input_grid = ColoredGrid(values=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 3, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0]
])

print("Input Grid:")
print_grid(input_grid)

output_grid = solve_ac3e2b04(input_grid)

print("\nOutput Grid:")
print_grid(output_grid)
