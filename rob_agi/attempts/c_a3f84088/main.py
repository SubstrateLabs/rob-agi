from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_a3f84088(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a nested outline pattern.
    
    The function does the following:
    1. Preserves the outer gray (5) outline from the input grid.
    2. Creates nested frames of alternating colors (red and gray) moving inward.
    3. Maintains a one-cell black (0) gap between each colored frame.
    4. Continues the pattern until reaching a 3x3 or smaller center.
    5. For a 3x3 center, fills it with gray (5).
    6. For a 2x2 or 1x1 center, fills it with black (0).
    7. Returns the transformed grid.
    """
    new_grid = input_grid.deep_copy()
    top, left, bottom, right = find_outer_boundary(new_grid)
    
    current_color = 2  # Start with red for the first inner frame
    previous_color = 5  # Gray

    while bottom - top > 2:
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
    if center_height == 3 and center_width == 3:
        fill_area(new_grid, top, left, bottom, right, 5)  # Fill with gray
    else:
        fill_area(new_grid, top, left, bottom, right, 0)  # Fill with black
    
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
