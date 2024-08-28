from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_4364c1c4.main import solve_4364c1c4

def print_grid(grid):
    for row in grid.values:
        print(''.join(str(cell) for cell in row))
    print()

# Test case from example_0
input_grid = ColoredGrid(values=[
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8],
    [8, 8, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8],
    [8, 8, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8],
    [8, 8, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8],
    [8, 8, 3, 3, 3, 3, 3, 3, 3, 8, 8, 8],
    [8, 8, 3, 3, 3, 3, 3, 3, 3, 8, 8, 8],
    [8, 8, 3, 3, 3, 3, 3, 3, 3, 8, 8, 8],
    [8, 8, 3, 3, 3, 3, 3, 3, 3, 8, 8, 8],
    [8, 8, 3, 3, 3, 8, 8, 3, 3, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
])

print("Input grid:")
print_grid(input_grid)

output_grid = solve_4364c1c4(input_grid)

print("Output grid:")
print_grid(output_grid)
