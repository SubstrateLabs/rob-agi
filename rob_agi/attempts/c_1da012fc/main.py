from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from itertools import cycle

def solve_1da012fc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by:
    1. Identifying the gray area and the unique colors within it.
    2. Finding the most common non-zero, non-gray color outside the gray area.
    3. Identifying connected regions of this target color.
    4. Sorting regions by their top-left coordinate.
    5. Transforming each region to a new color based on the sequence of colors in the gray area.
    6. Returning the transformed grid.
    """
    # Identify the gray area and colors within it
    gray_area = [(r, c) for r, row in enumerate(input_grid.values) for c, val in enumerate(row) if val == 5]
    colors_in_gray = sorted(set(input_grid.values[r][c] for r, c in gray_area if input_grid.values[r][c] != 5))
    new_colors = cycle(colors_in_gray)

    # Find the most common non-zero, non-gray color outside the gray area
    color_counts = Counter(cell for r, row in enumerate(input_grid.values) 
                           for c, cell in enumerate(row) 
                           if cell not in [0, 5] and (r, c) not in gray_area)
    target_color = color_counts.most_common(1)[0][0]

    # Find connected regions of the target color
    regions = input_grid.find_connected_regions(target_color)

    # Sort regions by top-left coordinate
    regions.sort(key=lambda region: min(region))

    # Create output grid
    output_grid = input_grid.deep_copy()

    # Transform regions
    for region in regions:
        new_color = next(new_colors)
        for row, col in region:
            output_grid.values[row][col] = new_color

    return output_grid
