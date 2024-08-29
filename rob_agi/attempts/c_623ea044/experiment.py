from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_623ea044.main import solve_623ea044

def print_grid(grid):
    for row in grid.values:
        print(''.join(str(cell) for cell in row))
    print()

# Test case 1: 7x7 grid with colored cell in the center
input_grid1 = ColoredGrid(values=[
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
])

# Test case 2: 15x15 grid with colored cell off-center
input_grid2 = ColoredGrid(values=[
    [0] * 15 for _ in range(15)
])
input_grid2.set_cell(5, 11, 7)

print("Test case 1 (7x7 grid):")
print("Input:")
print_grid(input_grid1)
print("Output:")
print_grid(solve_623ea044(input_grid1))

print("Test case 2 (15x15 grid):")
print("Input:")
print_grid(input_grid2)
print("Output:")
print_grid(solve_623ea044(input_grid2))
