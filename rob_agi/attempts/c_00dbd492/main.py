from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def find_red_rectangles(grid: ColoredGrid) -> List[Tuple[int, int, int, int]]:
    rectangles = []
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 2:  # Red pixel
                # Check if it's the top-left corner of a rectangle
                if (r == 0 or grid.get_cell(r-1, c) != 2) and (c == 0 or grid.get_cell(r, c-1) != 2):
                    # Find bottom-right corner
                    bottom = r
                    right = c
                    while bottom + 1 < rows and grid.get_cell(bottom + 1, c) == 2:
                        bottom += 1
                    while right + 1 < cols and grid.get_cell(r, right + 1) == 2:
                        right += 1
                    
                    # Verify all edges are red
                    if all(grid.get_cell(r, i) == 2 for i in range(c, right + 1)) and \
                       all(grid.get_cell(i, c) == 2 for i in range(r, bottom + 1)) and \
                       all(grid.get_cell(bottom, i) == 2 for i in range(c, right + 1)) and \
                       all(grid.get_cell(i, right) == 2 for i in range(r, bottom + 1)):
                        rectangles.append((r, c, bottom, right))
    return rectangles

def get_fill_color(width: int, height: int) -> int:
    if width == height == 5:
        return 8  # sky blue
    elif width == height == 7:
        return 4  # yellow
    elif width == height == 9:
        return 3  # green
    else:
        return 4  # default to yellow for other sizes

def fill_rectangle(grid: ColoredGrid, top: int, left: int, bottom: int, right: int, color: int) -> None:
    for r in range(top + 1, bottom):
        for c in range(left + 1, right):
            if grid.get_cell(r, c) == 0:  # Only fill black cells
                grid.set_cell(r, c, color)

def solve_00dbd492(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by identifying red rectangular outlines,
    filling them with appropriate colors based on size, and ensuring a central red pixel.
    
    The solution follows these steps:
    1. Find all red rectangular outlines in the input grid.
    2. For each rectangle:
       a. Determine the fill color based on its size.
       b. Fill the interior with the chosen color, preserving existing red pixels.
       c. Ensure there's a central red pixel, adding one if missing.
    3. Return the modified grid.
    """
    output_grid = input_grid.deep_copy()
    rectangles = find_red_rectangles(output_grid)
    
    for top, left, bottom, right in rectangles:
        width = right - left + 1
        height = bottom - top + 1
        fill_color = get_fill_color(width, height)
        fill_rectangle(output_grid, top, left, bottom, right, fill_color)
        
        # Ensure central red pixel
        center_r, center_c = (top + bottom) // 2, (left + right) // 2
        if output_grid.get_cell(center_r, center_c) == 0:
            output_grid.set_cell(center_r, center_c, 2)
    
    return output_grid
