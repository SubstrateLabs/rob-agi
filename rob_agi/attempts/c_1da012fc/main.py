from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_1da012fc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by:
    1. Identifying the gray area and extracting the color key within it.
    2. Identifying non-gray, non-black regions in the grid.
    3. Sorting regions by their top-left coordinate.
    4. Mapping each region to a new color based on the color key from the gray area.
    5. Transforming the grid by applying the new colors to each region.
    6. Returning the transformed grid.
    """
    # Step 1: Identify gray area and extract color key
    gray_area = [(r, c) for r, row in enumerate(input_grid.values) for c, val in enumerate(row) if val == 5]
    color_key = []
    for r, c in sorted(gray_area):
        color = input_grid.values[r][c]
        if color not in [0, 5] and color not in color_key:
            color_key.append(color)

    # Step 2: Identify non-gray, non-black regions
    regions = []
    visited = set()
    for r, row in enumerate(input_grid.values):
        for c, val in enumerate(row):
            if val not in [0, 5] and (r, c) not in visited:
                region = input_grid.find_connected_regions(val)[0]
                regions.append(region)
                visited.update(region)

    # Step 3: Sort regions
    regions.sort(key=lambda region: min(region))

    # Step 4: Create color mapping
    if len(regions) != len(color_key):
        raise ValueError("Number of regions doesn't match number of colors in gray area")
    color_mapping = {tuple(sorted(region)): new_color for region, new_color in zip(regions, color_key)}

    # Step 5: Transform the grid
    output_grid = input_grid.deep_copy()
    for region, new_color in color_mapping.items():
        for r, c in region:
            output_grid.values[r][c] = new_color

    # Step 6: Return the transformed grid
    return output_grid
