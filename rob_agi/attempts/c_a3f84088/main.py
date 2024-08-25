from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_a3f84088(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a nested outline pattern.
    
    The function does the following:
    1. Preserves the outer gray (5) outline from the input grid.
    2. Creates nested outlines of alternating colors (red and gray) moving inward.
    3. Handles the center area based on the remaining space size.
    4. Returns the transformed grid.

    The pattern consists of the original outer gray outline, followed by
    alternating red and gray outlines moving inward. The center
    pattern varies based on the remaining space size, with special
    handling for different grid dimensions.
    """
    new_grid = input_grid.deep_copy()
    top, left, bottom, right = find_outer_boundary(new_grid)
    
    current_color = 2  # Start with red for the first inner outline
    top += 1
    left += 1
    bottom -= 1
    right -= 1

    while bottom - top >= 1 and right - left >= 1:
        draw_outline(new_grid, top, left, bottom, right, current_color)
        
        if bottom - top <= 2 or right - left <= 2:
            break
        
        top += 1
        left += 1
        bottom -= 1
        right -= 1
        current_color = 7 - current_color  # Toggle between 2 (red) and 5 (gray)
    
    # Handle center area for small grids
    center_height = bottom - top + 1
    center_width = right - left + 1
    if center_height <= 2 and center_width <= 2:
        fill_area(new_grid, top, left, bottom, right, 5)
    
    return new_grid

def find_outer_boundary(grid: ColoredGrid) -> Tuple[int, int, int, int]:
    rows, cols = grid.get_dimensions()
    top = next(r for r in range(rows) if 5 in grid.values[r])
    bottom = next(r for r in range(rows-1, -1, -1) if 5 in grid.values[r])
    left = min(grid.values[r].index(5) for r in range(top, bottom+1))
    right = max(cols - 1 - grid.values[r][::-1].index(5) for r in range(top, bottom+1))
    return top, left, bottom, right

def draw_outline(grid: ColoredGrid, top: int, left: int, bottom: int, right: int, color: int):
    for c in range(left, right + 1):
        grid.values[top][c] = color
        grid.values[bottom][c] = color
    for r in range(top + 1, bottom):
        grid.values[r][left] = color
        grid.values[r][right] = color

def fill_area(grid: ColoredGrid, top: int, left: int, bottom: int, right: int, color: int):
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            grid.values[r][c] = color
