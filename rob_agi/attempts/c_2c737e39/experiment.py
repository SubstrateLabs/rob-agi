from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_2c737e39.main import solve_2c737e39

def print_grid(grid):
    for row in grid:
        print(' '.join(str(cell) if cell != 0 else '.' for cell in row))
    print()

# Test case where duplication should be down and to the left
input_grid = ColoredGrid(values=[
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 3, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 5, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

print("Input Grid:")
print_grid(input_grid.values)

result = solve_2c737e39(input_grid)

print("Output Grid:")
print_grid(result.values)

# Analyze the result
rows, cols = result.get_dimensions()
original_pattern = set()
duplicated_pattern = set()

for r in range(rows):
    for c in range(cols):
        if result.values[r][c] != 0:
            if r < 4:  # Assuming original pattern is in the top 4 rows
                original_pattern.add((r, c))
            else:
                duplicated_pattern.add((r, c))

print("Original pattern coordinates:", original_pattern)
print("Duplicated pattern coordinates:", duplicated_pattern)

if duplicated_pattern:
    min_r_orig = min(r for r, _ in original_pattern)
    min_c_orig = min(c for _, c in original_pattern)
    min_r_dup = min(r for r, _ in duplicated_pattern)
    min_c_dup = min(c for _, c in duplicated_pattern)
    
    shift_r = min_r_dup - min_r_orig
    shift_c = min_c_dup - min_c_orig
    
    print(f"Pattern shifted by: {shift_r} rows, {shift_c} columns")
else:
    print("No duplication occurred")
