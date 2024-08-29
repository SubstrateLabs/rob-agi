from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_f5b8619d.main import solve_f5b8619d

def print_grid(grid):
    for row in grid:
        print(' '.join(str(cell) for cell in row))
    print()

# Test with a more complex input
input_grid = ColoredGrid(values=[
    [2, 0, 3, 0, 1],
    [0, 4, 0, 5, 0],
    [6, 0, 7, 0, 9],
    [0, 1, 0, 2, 0],
    [3, 0, 4, 0, 5]
])

print("Input grid:")
print_grid(input_grid.values)

output_grid = solve_f5b8619d(input_grid)

print("Output grid:")
print_grid(output_grid.values)
