from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_bf32578f.main import solve_bf32578f

def print_grid(grid):
    for row in grid:
        print(' '.join(str(cell) for cell in row))
    print()

# Example 0 (10x10 grid)
input_grid = ColoredGrid(values=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 7, 0, 0, 0, 0, 0],
    [0, 0, 0, 7, 0, 0, 0, 0, 0, 0],
    [0, 0, 7, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 7, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 7, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 7, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

print("Input grid:")
print_grid(input_grid.values)

output_grid = solve_bf32578f(input_grid)

print("Output grid:")
print_grid(output_grid.values)
