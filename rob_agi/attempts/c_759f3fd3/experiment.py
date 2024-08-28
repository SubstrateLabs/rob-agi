from rob_agi.colored_grid import ColoredGrid

def create_test_grid(rows, cols):
    return ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

def fill_border(grid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        grid.values[r][0] = 4 if r % 2 == 0 else 0
        grid.values[r][-1] = 4 if r % 2 == 1 else 0
    for c in range(cols):
        grid.values[0][c] = 4 if c % 2 == 0 else 0
        grid.values[-1][c] = 4 if c % 2 == 1 else 0
    return grid

def print_grid(grid):
    for row in grid.values:
        print(' '.join(str(cell) for cell in row))
    print()

# Test with different grid sizes
test_sizes = [(5, 5), (10, 10), (20, 20)]

for rows, cols in test_sizes:
    print(f"Testing {rows}x{cols} grid:")
    grid = create_test_grid(rows, cols)
    filled_grid = fill_border(grid)
    print_grid(filled_grid)
