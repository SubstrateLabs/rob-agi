from rob_agi.colored_grid import ColoredGrid

def check_region(grid, start_row, end_row, start_col, end_col):
    return any(grid.get_cell(r, c) == 1 
               for r in range(start_row, end_row) 
               for c in range(start_col, end_col))

# Example 2 input
input_grid = ColoredGrid(values=[
    [1, 1, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 2, 2],
    [1, 1, 0, 2, 2]
])

# Check each quadrant
top_left = check_region(input_grid, 0, 3, 0, 3)
top_right = check_region(input_grid, 0, 3, 2, 5)
bottom_left = check_region(input_grid, 2, 5, 0, 3)
bottom_right = check_region(input_grid, 2, 5, 2, 5)

print(f"Top-left: {top_left}")
print(f"Top-right: {top_right}")
print(f"Bottom-left: {bottom_left}")
print(f"Bottom-right: {bottom_right}")

# Check specific cells in the top-right quadrant
for r in range(3):
    for c in range(2, 5):
        print(f"Cell at ({r}, {c}): {input_grid.get_cell(r, c)}")
