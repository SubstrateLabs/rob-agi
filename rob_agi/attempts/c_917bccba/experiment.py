from rob_agi.colored_grid import ColoredGrid

def analyze_grid(grid):
    rows, cols = len(grid), len(grid[0])
    colors = set()
    shape_left, shape_right = cols, 0
    shape_top, shape_bottom = rows, 0
    cross_vertical, cross_horizontal = None, None

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                colors.add(grid[r][c])
                if cross_vertical is None and all(grid[i][c] != 0 for i in range(rows)):
                    cross_vertical = c
                if cross_horizontal is None and all(grid[r][i] != 0 for i in range(cols)):
                    cross_horizontal = r
                if grid[r][c] != grid[cross_horizontal][cross_vertical]:
                    shape_left = min(shape_left, c)
                    shape_right = max(shape_right, c)
                    shape_top = min(shape_top, r)
                    shape_bottom = max(shape_bottom, r)

    shape_color = next(color for color in colors if color != grid[cross_horizontal][cross_vertical])
    cross_color = grid[cross_horizontal][cross_vertical]

    print(f"Shape Color: {shape_color}")
    print(f"Cross Color: {cross_color}")
    print(f"Shape Dimensions: {shape_right - shape_left + 1}x{shape_bottom - shape_top + 1}")
    print(f"Shape Position: Top={shape_top}, Left={shape_left}")
    print(f"Cross Position: Vertical={cross_vertical}, Horizontal={cross_horizontal}")

# Example grids
grids = [
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [3, 3, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [8, 8, 1, 8, 8, 8, 8, 1, 8, 8, 8, 8],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0]
    ]
]

for i, grid in enumerate(grids):
    print(f"\nAnalyzing Grid {i + 1}:")
    analyze_grid(grid)
