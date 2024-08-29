from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_8eb1be9a.main import solve_8eb1be9a

def find_first_non_zero_row(grid):
    return next((i for i, row in enumerate(grid) if any(row)), -1)

def extract_pattern(grid, start_row):
    pattern = []
    for i in range(start_row, min(start_row + 3, len(grid))):
        pattern.append(grid[i][:])
    while len(pattern) < 3:
        pattern.append(pattern[-1][:] if pattern else [0] * len(grid[0]))
    return pattern

def align_pattern(pattern):
    width = len(pattern[0])
    aligned_pattern = []
    for row in pattern:
        first_non_zero = next((i for i, x in enumerate(row) if x != 0), 0)
        aligned_row = row[first_non_zero:] + row[:first_non_zero]
        aligned_pattern.append(aligned_row)
    return aligned_pattern

# Test case 0 input
input_grid = ColoredGrid(values=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

# Find the first non-zero row
start_row = find_first_non_zero_row(input_grid.values)
print(f"First non-zero row: {start_row}")

# Extract the pattern
pattern = extract_pattern(input_grid.values, start_row)
print("Extracted pattern:")
for row in pattern:
    print(row)

# Align the pattern
aligned_pattern = align_pattern(pattern)
print("\nAligned pattern:")
for row in aligned_pattern:
    print(row)

# Solve the grid
output_grid = solve_8eb1be9a(input_grid)

print("\nFirst 5 rows of the output grid:")
for row in output_grid.values[:5]:
    print(row)

print("\nExpected first row:")
print([0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0])
