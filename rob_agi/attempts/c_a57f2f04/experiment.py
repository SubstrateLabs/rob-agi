from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_a57f2f04.main import generate_pattern, apply_pattern

def print_grid(grid):
    for row in grid:
        print(' '.join(str(cell) for cell in row))
    print()

# Test pattern generation
red_pattern = generate_pattern(2)
print("Red pattern:")
print_grid(red_pattern)

# Create a test grid
test_grid = ColoredGrid(values=[
    [8, 8, 8, 8, 8, 8, 8],
    [8, 2, 2, 2, 2, 2, 8],
    [8, 2, 2, 2, 2, 2, 8],
    [8, 2, 2, 2, 2, 2, 8],
    [8, 2, 2, 2, 2, 2, 8],
    [8, 2, 2, 2, 2, 2, 8],
    [8, 8, 8, 8, 8, 8, 8]
])

# Apply the pattern
region = [(r, c) for r in range(1, 6) for c in range(1, 6)]
apply_pattern(test_grid, red_pattern, region)

print("Resulting grid after applying red pattern:")
print_grid(test_grid.values)

# Test pattern application with offset
offset_test_grid = ColoredGrid(values=[
    [8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 2, 2, 2, 2, 2, 8],
    [8, 8, 2, 2, 2, 2, 2, 8],
    [8, 8, 2, 2, 2, 2, 2, 8],
    [8, 8, 2, 2, 2, 2, 2, 8],
    [8, 8, 2, 2, 2, 2, 2, 8],
    [8, 8, 8, 8, 8, 8, 8, 8]
])

offset_region = [(r, c) for r in range(1, 6) for c in range(2, 7)]
apply_pattern(offset_test_grid, red_pattern, offset_region)

print("Resulting grid after applying red pattern with offset:")
print_grid(offset_test_grid.values)
