from rob_agi.colored_grid import ColoredGrid

def find_largest_l_shape(grid: ColoredGrid) -> tuple:
    rows, cols = grid.get_dimensions()
    best_l = (0, (-1, -1), 0, 0)  # (size, (row, col), vertical_length, horizontal_length)

    for r in range(rows - 1, -1, -1):
        for c in range(cols - 1, -1, -1):
            if grid.get_cell(r, c) == 0:  # If it's a black cell
                vertical_length = 1
                while r - vertical_length >= 0 and grid.get_cell(r - vertical_length, c) == 0:
                    vertical_length += 1
                vertical_length -= 1

                horizontal_length = 1
                while c - horizontal_length >= 0 and grid.get_cell(r, c - horizontal_length) == 0:
                    horizontal_length += 1
                horizontal_length -= 1

                l_size = vertical_length + horizontal_length - 1
                if l_size > best_l[0] or (l_size == best_l[0] and (r, c) > best_l[1]):
                    best_l = (l_size, (r, c), vertical_length, horizontal_length)

    return best_l

# Test cases
test_grids = [
    [
        [0, 0, 8, 8],
        [0, 8, 8, 8],
        [0, 0, 0, 8],
        [8, 8, 8, 8]
    ],
    [
        [8, 8, 8, 8],
        [8, 0, 0, 8],
        [8, 0, 8, 8],
        [8, 0, 0, 0]
    ],
    [
        [0, 0, 0, 0],
        [0, 8, 8, 8],
        [0, 8, 8, 8],
        [0, 0, 0, 0]
    ]
]

for i, grid_values in enumerate(test_grids):
    grid = ColoredGrid(values=grid_values)
    result = find_largest_l_shape(grid)
    print(f"Test case {i + 1}:")
    print(f"Grid:")
    for row in grid_values:
        print(row)
    print(f"Largest L-shape: {result}")
    print()

# Edge case: grid with no black cells
all_sky_grid = ColoredGrid(values=[[8 for _ in range(4)] for _ in range(4)])
result = find_largest_l_shape(all_sky_grid)
print("Edge case: Grid with no black cells")
print(f"Largest L-shape: {result}")
print()

# Edge case: grid with only black cells
all_black_grid = ColoredGrid(values=[[0 for _ in range(4)] for _ in range(4)])
result = find_largest_l_shape(all_black_grid)
print("Edge case: Grid with only black cells")
print(f"Largest L-shape: {result}")
