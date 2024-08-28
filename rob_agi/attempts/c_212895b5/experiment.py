from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_212895b5.main import solve_212895b5, propagate_color, RED, YELLOW, RED_DIR, YELLOW_DIR

def print_grid(grid):
    for row in grid:
        print(' '.join(str(cell) for cell in row))
    print()

# Test case
input_grid = ColoredGrid(values=[
    [0, 0, 5, 0, 0],
    [0, 0, 0, 0, 5],
    [0, 5, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [5, 0, 0, 0, 0]
])

print("Input grid:")
print_grid(input_grid.values)

result = solve_212895b5(input_grid)

print("Output grid:")
print_grid(result.values)

# Test propagate_color function
test_grid = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 5, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]

print("Testing propagate_color for RED:")
propagate_color(test_grid, 2, 2, RED, RED_DIR, 4)
print_grid(test_grid)

test_grid = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 5, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]

print("Testing propagate_color for YELLOW:")
propagate_color(test_grid, 2, 2, YELLOW, YELLOW_DIR, 4)
print_grid(test_grid)
