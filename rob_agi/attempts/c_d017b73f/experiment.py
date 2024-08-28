from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_d017b73f.main import solve_d017b73f

def print_grid(grid):
    for row in grid:
        print(' '.join(str(cell) for cell in row))
    print()

# Test case from the visual descriptions
input_grid = ColoredGrid(values=[
    [0, 0, 0, 4, 4, 0, 0, 1, 0, 2, 2],
    [1, 1, 0, 0, 0, 0, 3, 0, 0, 2, 2],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

print("Input grid:")
print_grid(input_grid.values)

output_grid = solve_d017b73f(input_grid)

print("Output grid:")
print_grid(output_grid.values)
