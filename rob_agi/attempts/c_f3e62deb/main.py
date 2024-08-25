from rob_agi.colored_grid import ColoredGrid
from typing import Tuple

def solve_f3e62deb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by moving a 3x3 hollow square shape.
    
    The function identifies the 3x3 hollow square in the input grid and moves it
    to the nearest available edge in the following priority order:
    1. Right edge (if not already there)
    2. Top edge (if not at right edge)
    3. Left edge (if not at top edge)
    4. Bottom edge (if not at left edge)
    
    The shape maintains its vertical position when moving horizontally and
    its horizontal position when moving vertically.
    """
    def find_hollow_square(grid: ColoredGrid) -> Tuple[int, int, int]:
        for color in range(1, 10):  # Check all non-black colors
            regions = grid.find_connected_regions(color)
            for region in regions:
                if len(region) == 8:  # A hollow square has 8 colored cells
                    top = min(r for r, _ in region)
                    left = min(c for _, c in region)
                    if grid.extract_subgrid(top, left, 3, 3).values == [
                        [color, color, color],
                        [color, 0, color],
                        [color, color, color]
                    ]:
                        return top, left, color
        return -1, -1, -1

    top, left, color = find_hollow_square(input_grid)
    if top == -1:
        return input_grid  # No valid square found, return input grid unchanged

    # Determine new position
    if left < 7:
        new_left, new_top = 7, top  # Move to right edge
    elif top > 0:
        new_left, new_top = left, 0  # Move to top edge
    elif left > 0:
        new_left, new_top = 0, top  # Move to left edge
    else:
        new_left, new_top = left, 7  # Move to bottom edge

    # Create new grid with moved square
    new_grid = ColoredGrid(values=[[0 for _ in range(10)] for _ in range(10)])
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue  # Skip the center (keep it black)
            new_grid.values[new_top + i][new_left + j] = color

    return new_grid
