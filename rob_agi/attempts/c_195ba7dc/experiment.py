from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_195ba7dc.main import solve_195ba7dc

def print_grid_section(grid, start_col, end_col):
    for row in grid.values:
        print(row[start_col:end_col])
    print()

# Example input grid
input_grid = ColoredGrid(values=[
    [0, 7, 7, 0, 7, 7, 2, 7, 0, 0, 0, 0, 7],
    [7, 0, 0, 0, 0, 7, 2, 7, 0, 0, 7, 7, 0],
    [7, 0, 7, 7, 0, 7, 2, 7, 0, 0, 7, 0, 0],
    [0, 7, 0, 0, 0, 0, 2, 7, 0, 7, 0, 7, 0],
    [7, 7, 0, 7, 7, 0, 2, 0, 7, 0, 0, 7, 0]
])

output_grid = solve_195ba7dc(input_grid)

print("Input grid right section (columns 7-12):")
print_grid_section(input_grid, 7, 13)

print("Output grid right section (columns 3-5):")
print_grid_section(output_grid, 3, 6)

print("Expected output for right section based on transformation rules:")
for row in input_grid.values:
    expected_row = [
        1 if 7 in row[7:9] else 0,
        1 if 7 in row[9:11] else 0,
        1 if 7 in row[11:13] else 0
    ]
    print(expected_row)
