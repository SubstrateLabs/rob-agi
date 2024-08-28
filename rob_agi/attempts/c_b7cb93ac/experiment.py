from rob_agi.colored_grid import ColoredGrid
from collections import Counter
import itertools

def analyze_grid(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    color_counts = Counter(color for row in grid.values for color in row if color != 0)
    print(f"Color distribution: {color_counts}")

    # Analyze horizontal patterns
    max_horizontal = 0
    horizontal_color = 0
    for row in range(rows):
        for color, group in itertools.groupby(grid.values[row]):
            if color != 0:
                length = len(list(group))
                if length > max_horizontal:
                    max_horizontal = length
                    horizontal_color = color
    print(f"Longest horizontal line: color {horizontal_color}, length {max_horizontal}")

    # Analyze vertical patterns
    max_vertical = 0
    vertical_color = 0
    for col in range(cols):
        column = [grid.values[row][col] for row in range(rows)]
        for color, group in itertools.groupby(column):
            if color != 0:
                length = len(list(group))
                if length > max_vertical:
                    max_vertical = length
                    vertical_color = color
    print(f"Longest vertical line: color {vertical_color}, length {max_vertical}")

    # Check for sky blue (8)
    sky_blue_coords = [(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 8]
    print(f"Sky blue (8) coordinates: {sky_blue_coords}")

# Test with example grids
example_grids = [
    ColoredGrid(values=[
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]),
    ColoredGrid(values=[
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0],
        [0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]),
    ColoredGrid(values=[
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
        [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 8, 0, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
]

for i, grid in enumerate(example_grids):
    print(f"\nAnalyzing Example Grid {i}:")
    analyze_grid(grid)
