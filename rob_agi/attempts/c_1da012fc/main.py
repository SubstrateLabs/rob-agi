from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from itertools import cycle

def solve_1da012fc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by:
    1. Identifying the gray area and extracting the color sequence within it.
    2. Finding the most common non-zero, non-gray color outside the gray area.
    3. Identifying connected regions of this target color.
    4. Sorting regions by their top-left coordinate.
    5. Transforming each region to a new color based on the cyclic sequence of colors from the gray area.
    6. Returning the transformed grid.
    """
    # Step 1: Identify the gray area and extract color sequence
    gray_area = [(r, c) for r, row in enumerate(input_grid.values) for c, val in enumerate(row) if val == 5]
    color_sequence = []
    for r, c in sorted(gray_area):
        color = input_grid.values[r][c]
        if color != 5 and color not in color_sequence:
            color_sequence.append(color)
    color_cycle = cycle(color_sequence)

    # Step 2: Find the most common non-zero, non-gray color outside the gray area
    color_counts = Counter(cell for r, row in enumerate(input_grid.values) 
                           for c, cell in enumerate(row) 
                           if cell not in [0, 5] and (r, c) not in gray_area)
    target_color = color_counts.most_common(1)[0][0]

    # Step 3: Find connected regions of the target color
    regions = input_grid.find_connected_regions(target_color)

    # Step 4: Sort regions by top-left coordinate
    regions.sort(key=lambda region: min(region))

    # Step 5: Transform regions
    output_grid = input_grid.deep_copy()
    for region in regions:
        new_color = next(color_cycle)
        for r, c in region:
            output_grid.values[r][c] = new_color

    # Step 6: Return the transformed grid
    return output_grid
