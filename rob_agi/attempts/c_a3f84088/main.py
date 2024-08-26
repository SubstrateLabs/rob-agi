from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_a3f84088(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a nested outline pattern.
    
    The function does the following:
    1. Preserves the outer gray (5) outline from the input grid.
    2. Creates nested frames of alternating colors (red and gray) moving inward.
    3. Handles the center area based on the remaining space size:
       - For 2x2 or smaller center, fills it with the current color.
       - For 3x3, draws an outline with the current color, leaving the center black.
       - For 4x4 and larger, draws an outline with the current color and fills the inside with the previous color.
    4. Maintains a one-cell black (0) gap between each colored frame.
    5. Returns the transformed grid.
    """
    new_grid = input_grid.deep_copy()
    top, left, bottom, right = find_outer_boundary(new_grid)
    
    current_color = 2  # Start with red for the first inner frame
    previous_color = 5  # Gray

    while bottom - top > 2 and right - left > 2:
        # Draw current color frame
        draw_outline(new_grid, top+1, left+1, bottom-1, right-1, current_color)
        
        # Shrink the working area
        top += 2
        left += 2
        bottom -= 2
        right -= 2

        # Swap colors
        current_color, previous_color = previous_color, current_color

    # Handle center area
    center_height = bottom - top + 1
    center_width = right - left + 1
    if center_height <= 2 and center_width <= 2:
        fill_area(new_grid, top, left, bottom, right, current_color)
    elif center_height == 3 and center_width == 3:
        draw_outline(new_grid, top, left, bottom, right, current_color)
    else:
        draw_outline(new_grid, top, left, bottom, right, current_color)
        fill_area(new_grid, top+1, left+1, bottom-1, right-1, previous_color)
    
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
