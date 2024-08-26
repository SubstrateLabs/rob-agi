from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_1a6449f1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Extracts a subgrid from the input grid based on the following rules:
    1. Find the largest rectangular region that doesn't contain the most common non-black color.
    2. This region should be surrounded by the most common non-black color on at least two sides.
    3. Extract this region as the output subgrid.
    """
    # Find the most common non-black color
    color_counts = {color: count for color, count in input_grid.get_color_frequencies().items() if color != 0}
    most_common_color = max(color_counts, key=color_counts.get)

    # Find the largest region not containing the most common color
    rows, cols = input_grid.get_dimensions()
    max_area = 0
    max_region = None

    for top in range(rows):
        for left in range(cols):
            for bottom in range(top, rows):
                for right in range(left, cols):
                    if all(input_grid.get_cell(r, c) != most_common_color 
                           for r in range(top, bottom+1) 
                           for c in range(left, right+1)):
                        area = (bottom - top + 1) * (right - left + 1)
                        if area > max_area:
                            # Check if surrounded by most common color on at least two sides
                            sides_surrounded = 0
                            if top > 0 and all(input_grid.get_cell(top-1, c) == most_common_color for c in range(left, right+1)):
                                sides_surrounded += 1
                            if bottom < rows-1 and all(input_grid.get_cell(bottom+1, c) == most_common_color for c in range(left, right+1)):
                                sides_surrounded += 1
                            if left > 0 and all(input_grid.get_cell(r, left-1) == most_common_color for r in range(top, bottom+1)):
                                sides_surrounded += 1
                            if right < cols-1 and all(input_grid.get_cell(r, right+1) == most_common_color for r in range(top, bottom+1)):
                                sides_surrounded += 1
                            
                            if sides_surrounded >= 2:
                                max_area = area
                                max_region = (top, left, bottom, right)

    if max_region:
        top, left, bottom, right = max_region
        return input_grid.extract_subgrid(top, left, bottom-top+1, right-left+1)
    else:
        return ColoredGrid(values=[[0]])  # Return a 1x1 black grid if no suitable region found
