from rob_agi.colored_grid import ColoredGrid

def solve_aab50785(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the aab50785 challenge by finding the largest rectangular region in the grid
    that is bordered by continuous 8's on exactly two sides and doesn't contain any 8's inside.
    
    The function iterates through all possible rectangles in the grid, validates them
    against the criteria, and returns the largest valid rectangle as a new ColoredGrid.
    
    The algorithm works as follows:
    1. Iterate through all possible rectangles in the grid.
    2. For each rectangle, check if it's bordered by continuous 8's on exactly two sides.
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
    max_area = 0
    best_rectangle = None

    for top in range(rows):
        for left in range(cols):
            for bottom in range(top, rows):
                for right in range(left, cols):
                    if is_bordered_by_eights(input_grid, top, left, bottom, right) and \
                       contains_no_eights(input_grid, top, left, bottom, right):
                        area = (bottom - top + 1) * (right - left + 1)
                        if area > max_area:
                            max_area = area
                            best_rectangle = (top, left, bottom, right)

    if best_rectangle is not None:
        top, left, bottom, right = best_rectangle
        return input_grid.extract_subgrid(top, left, bottom - top + 1, right - left + 1)
    else:
        return ColoredGrid(values=[[]])

def is_bordered_by_eights(grid: ColoredGrid, top: int, left: int, bottom: int, right: int) -> bool:
    """Check if the rectangle is bordered by continuous 8's on exactly two sides."""
    sides_with_eights = 0
    
    # Check top side
    if top > 0 and all(grid.get_cell(top-1, c) == 8 for c in range(left, right+1)):
        sides_with_eights += 1
    
    # Check bottom side
    if bottom < grid.num_rows - 1 and all(grid.get_cell(bottom+1, c) == 8 for c in range(left, right+1)):
        sides_with_eights += 1
    
    # Check left side
    if left > 0 and all(grid.get_cell(r, left-1) == 8 for r in range(top, bottom+1)):
        sides_with_eights += 1
    
    # Check right side
    if right < grid.num_cols - 1 and all(grid.get_cell(r, right+1) == 8 for r in range(top, bottom+1)):
        sides_with_eights += 1
    
    return sides_with_eights == 2

def contains_no_eights(grid: ColoredGrid, top: int, left: int, bottom: int, right: int) -> bool:
    """Verify that the rectangle doesn't contain any 8's inside."""
    return all(grid.get_cell(r, c) != 8 for r in range(top, bottom+1) for c in range(left, right+1))
