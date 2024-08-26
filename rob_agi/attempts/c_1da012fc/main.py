from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from itertools import cycle

def solve_1da012fc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by:
    1. Identifying the most common non-zero, non-gray color.
    2. Finding connected regions of this color.
    3. Sorting regions by their top-left coordinate.
    4. Transforming each region to a new color in the sequence [red, green, yellow, magenta].
    5. Returning the transformed grid.
    """
    # Find the most common non-zero, non-gray color
    color_counts = Counter(cell for row in input_grid.values for cell in row if cell not in [0, 5])
    target_color = color_counts.most_common(1)[0][0]

    # Find connected regions of the target color
    regions = input_grid.find_connected_regions(target_color)

    # Sort regions by top-left coordinate
    regions.sort(key=lambda region: min(region))

    # Prepare color transformation sequence
    new_colors = cycle([2, 3, 4, 6])  # red, green, yellow, magenta

    # Create output grid
    output_grid = input_grid.deep_copy()

    # Transform regions
    for region in regions:
        new_color = next(new_colors)
        for row, col in region:
            output_grid.values[row][col] = new_color

    return output_grid
