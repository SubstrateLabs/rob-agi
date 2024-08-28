from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_bbb1b8b6.main import solve_bbb1b8b6, identify_shapes, expand_shape

def print_grid(grid):
    for row in grid:
        print(" ".join(str(cell) for cell in row))
    print()

# Example input grid (example 3 from the test cases)
input_grid = ColoredGrid(values=[
    [1, 1, 1, 1, 5, 0, 0, 0, 0],
    [1, 0, 0, 1, 5, 0, 6, 6, 0],
    [1, 0, 0, 1, 5, 0, 6, 6, 0],
    [1, 1, 1, 1, 5, 0, 0, 0, 0]
])

print("Input Grid:")
print_grid(input_grid.values)

# Extract right half
right_half = [row[5:9] for row in input_grid.values]

print("Right Half:")
print_grid(right_half)

# Identify shapes in the right half
shapes = identify_shapes(right_half)

print("Identified Shapes:")
for color, shape in shapes:
    print(f"Color: {color}, Shape: {shape}")

# Create a copy of the left half to work with
result_grid = [row[:4] for row in input_grid.values]

print("Initial Left Half:")
print_grid(result_grid)

# Expand each shape
for color, shape in shapes:
    expand_shape(result_grid, shape, color)
    print(f"After expanding shape with color {color}:")
    print_grid(result_grid)

print("Final Result:")
print_grid(result_grid)
