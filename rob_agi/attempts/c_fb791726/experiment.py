from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_fb791726.main import solve_fb791726

def print_grid(grid):
    for row in grid.values:
        print(' '.join(str(cell) for cell in row))
    print()

# Test case
input_grid = ColoredGrid(values=[
    [1, 0, 0, 0],
    [0, 2, 0, 0],
    [0, 0, 3, 0],
    [0, 0, 0, 4]
])

print("Input grid:")
print_grid(input_grid)

output_grid = solve_fb791726(input_grid)

print("Output grid:")
print_grid(output_grid)

# Verify green separator column placement
n = len(input_grid.values)
print(f"Green separator column should be at index {n-1}")
print(f"Is green separator column correct? {all(row[n-1] == 3 for row in output_grid.values)}")

# Verify quadrant transformations
def check_quadrant(quad_name, start_row, start_col, end_row, end_col, expected_values):
    print(f"\nChecking {quad_name} quadrant:")
    for i, row in enumerate(range(start_row, end_row)):
        for j, col in enumerate(range(start_col, end_col)):
            actual = output_grid.values[row][col]
            expected = expected_values[i][j]
            print(f"Position ({row}, {col}): Expected {expected}, Actual {actual}")

# Define expected values for each quadrant
top_left = [[1, 0], [0, 2]]
top_right = [[0, 0], [3, 0]]
bottom_left = [[0, 3], [0, 0]]
bottom_right = [[3, 0], [0, 4]]

check_quadrant("Top-left", 0, 0, 2, 2, top_left)
check_quadrant("Top-right", 0, n, 2, n+2, top_right)
check_quadrant("Bottom-left", n, 0, n+2, 2, bottom_left)
check_quadrant("Bottom-right", n, n, n+2, n+2, bottom_right)
