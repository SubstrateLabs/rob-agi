from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_a3f84088(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a nested outline pattern.
    
    The function does the following:
    1. Preserves the outer gray (5) outline from the input grid.
    2. Creates nested frames of alternating colors (red and gray) moving inward.
    3. Handles the center area based on the remaining space size:
       - For 3x3 or smaller center, fills it with the appropriate pattern.
       - For 4x4 or 5x5 centers, applies a specific pattern.
    4. Returns the transformed grid.

    The pattern consists of the original outer gray outline, followed by
    alternating red and gray frames moving inward, with each frame being
    separated by a one-cell gap of the previous color.
    """
    new_grid = input_grid.deep_copy()
    top, left, bottom, right = find_outer_boundary(new_grid)
    
    current_color = 2  # Start with red for the first inner frame
    previous_color = 5  # Gray

    while bottom - top > 4 and right - left > 4:
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
    if center_height <= 3 and center_width <= 3:
        fill_area(new_grid, top, left, bottom, right, current_color)
        if center_height == 3 and center_width == 3:
            new_grid.values[top+1][left+1] = previous_color
    elif center_height == 4 and center_width == 4:
        fill_area(new_grid, top, left, bottom, right, current_color)
    elif center_height == 5 and center_width == 5:
        draw_outline(new_grid, top, left, bottom, right, current_color)
        fill_area(new_grid, top+1, left+1, bottom-1, right-1, previous_color)
        new_grid.values[top+2][left+2] = current_color
    
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
