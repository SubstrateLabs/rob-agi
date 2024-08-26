from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional

def solve_aab50785(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the aab50785 challenge by finding the largest rectangular region in the grid
    that is bordered by 8's on exactly two sides and doesn't contain any 8's inside.
    
    The function iterates through all possible rectangles in the grid, validates them
    against the criteria, and returns the largest valid rectangle as a new ColoredGrid.
    
    The solution considers rectangles that are bordered by 8's on exactly two sides,
    which can be any two sides (not necessarily opposite). This approach captures
    regions that are part of larger structures in the grid while ensuring they meet
    the specific criteria.
    
    The algorithm works as follows:
    1. Iterate through all possible rectangles in the grid.
    2. For each rectangle, check if it's bordered by 8's on exactly two sides.
    3. If so, verify that it doesn't contain any 8's inside.
    4. If both conditions are met, compare its area to the largest found so far.
    5. Keep track of the largest valid rectangle.
    6. Finally, extract and return the largest valid rectangle found.

    Args:
    input_grid (ColoredGrid): The input grid to process

    Returns:
    ColoredGrid: The largest valid region extracted from the input grid,
                 or an empty grid if no valid region is found.
    """
    rows, cols = input_grid.get_dimensions()
    
    largest_valid_rect = None
    largest_area = 0

    for top in range(rows):
        for left in range(cols):
            for bottom in range(top, rows):
                for right in range(left, cols):
                    if is_valid_region(input_grid, top, left, bottom, right):
                        area = (bottom - top + 1) * (right - left + 1)
                        if area > largest_area:
                            largest_area = area
                            largest_valid_rect = (top, left, bottom, right)
    
    if largest_valid_rect is None:
        return ColoredGrid(values=[[]])
    
    top, left, bottom, right = largest_valid_rect
    extracted = input_grid.extract_subgrid(top, left, bottom - top + 1, right - left + 1)
    
    return extracted

def is_valid_region(grid: ColoredGrid, top: int, left: int, bottom: int, right: int) -> bool:
    """Check if a region is valid according to the criteria."""
    # Check if bordered by 8's on exactly two sides
    sides_with_eights = 0
    top_has_eight = any(grid.get_cell(top, c) == 8 for c in range(left, right+1))
    bottom_has_eight = any(grid.get_cell(bottom, c) == 8 for c in range(left, right+1))
    left_has_eight = any(grid.get_cell(r, left) == 8 for r in range(top, bottom+1))
    right_has_eight = any(grid.get_cell(r, right) == 8 for r in range(top, bottom+1))
    
    sides_with_eights = sum([top_has_eight, bottom_has_eight, left_has_eight, right_has_eight])
    
    if sides_with_eights != 2:
        return False
    
    # Check if there are no 8's inside the region
    for r in range(top+1, bottom):
        for c in range(left+1, right):
            if grid.get_cell(r, c) == 8:
                return False
    
    return True
