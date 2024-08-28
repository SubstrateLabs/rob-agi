from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_b457fec5.main import solve_b457fec5

def print_grid(grid):
    for row in grid.values:
        print(' '.join(str(cell) for cell in row))
    print()

# Test case 1: Left-to-right fill
input_grid1 = ColoredGrid(values=[
    [0, 0, 0, 0, 0],
    [0, 1, 2, 3, 0],
    [0, 5, 5, 5, 0],
    [0, 5, 5, 5, 0],
    [0, 0, 0, 0, 0]
])

# Test case 2: Right-to-left fill
input_grid2 = ColoredGrid(values=[
    [0, 0, 0, 0, 0],
    [0, 0, 1, 2, 0],
    [0, 5, 5, 5, 0],
    [0, 5, 5, 5, 0],
    [0, 0, 0, 0, 0]
])

# Test case 3: Disconnected gray areas
input_grid3 = ColoredGrid(values=[
    [0, 0, 0, 0, 0],
    [0, 1, 2, 3, 0],
    [0, 5, 0, 5, 0],
    [0, 5, 5, 5, 0],
    [0, 0, 0, 0, 0]
])

print("Test case 1 (Left-to-right fill):")
print("Input:")
print_grid(input_grid1)
print("Output:")
print_grid(solve_b457fec5(input_grid1))

print("Test case 2 (Right-to-left fill):")
print("Input:")
print_grid(input_grid2)
print("Output:")
print_grid(solve_b457fec5(input_grid2))

print("Test case 3 (Disconnected gray areas):")
print("Input:")
print_grid(input_grid3)
print("Output:")
print_grid(solve_b457fec5(input_grid3))
