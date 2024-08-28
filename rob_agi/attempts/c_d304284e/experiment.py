from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_d304284e.main import solve_d304284e

def print_grid(grid):
    for row in grid.values:
        print(' '.join(str(cell) for cell in row))

# Create a sample input grid
input_grid = ColoredGrid(values=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 7, 7, 7, 0, 0, 0, 0],
    [0, 0, 0, 7, 0, 7, 0, 0, 0, 0],
    [0, 0, 0, 7, 7, 7, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

print("Input Grid:")
print_grid(input_grid)

# Solve the challenge
output_grid = solve_d304284e(input_grid)

print("\nOutput Grid:")
print_grid(output_grid)
