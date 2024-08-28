from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_dc2aa30b.main import solve_dc2aa30b
import random

def print_grid(grid):
    for row in grid.values:
        print(' '.join(str(cell) for cell in row))
    print()

# Set a fixed seed for reproducibility
random.seed(42)

# Test case from test_dc2aa30b_example_0
input_grid = ColoredGrid(values=[
    [2, 2, 1, 0, 2, 2, 2, 0, 1, 2, 1],
    [1, 2, 2, 0, 2, 2, 2, 0, 1, 1, 2],
    [2, 2, 2, 0, 1, 2, 2, 0, 2, 1, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 1, 0, 2, 1, 2, 0, 2, 2, 2],
    [1, 2, 2, 0, 1, 2, 1, 0, 2, 2, 2],
    [2, 1, 2, 0, 2, 2, 1, 0, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [1, 2, 1, 0, 1, 1, 1, 0, 2, 1, 1],
    [1, 2, 1, 0, 1, 2, 1, 0, 1, 1, 2]
])

print("Input Grid:")
print_grid(input_grid)

output_grid = solve_dc2aa30b(input_grid)

print("Output Grid:")
print_grid(output_grid)

# Count colors in each section
def count_colors(grid, start_row, end_row):
    blue_count = sum(cell == 1 for row in grid.values[start_row:end_row] for cell in row if cell != 0)
    red_count = sum(cell == 2 for row in grid.values[start_row:end_row] for cell in row if cell != 0)
    return blue_count, red_count

top_blue, top_red = count_colors(output_grid, 0, 3)
middle_blue, middle_red = count_colors(output_grid, 4, 7)
bottom_blue, bottom_red = count_colors(output_grid, 8, 11)

print(f"Top section: {top_blue} blue, {top_red} red")
print(f"Middle section: {middle_blue} blue, {middle_red} red")
print(f"Bottom section: {bottom_blue} blue, {bottom_red} red")
