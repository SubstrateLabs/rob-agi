from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_23b5c85d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Finds the smallest isolated rectangle of non-zero color in the input grid.
    
    The function identifies all non-zero colored regions, determines if they are
    rectangles and isolated (not part of a larger region of the same color),
    and returns the smallest such rectangle.

    Args:
    input_grid (ColoredGrid): The input grid to analyze.

    Returns:
    ColoredGrid: A new grid containing only the smallest isolated rectangle.
    """
    def is_rectangle(color: int, top: int, left: int, bottom: int, right: int) -> bool:
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                if input_grid.get_cell(r, c) != color:
                    return False
        return True

    def is_isolated(color: int, top: int, left: int, bottom: int, right: int) -> bool:
        rows, cols = input_grid.get_dimensions()
        for r in range(max(0, top - 1), min(rows, bottom + 2)):
            for c in range(max(0, left - 1), min(cols, right + 2)):
                if (r < top or r > bottom or c < left or c > right) and input_grid.get_cell(r, c) == color:
                    return False
        return True

    rows, cols = input_grid.get_dimensions()
    rectangles = []

    for color in range(1, 10):  # Assuming colors are 1-9
        for top in range(rows):
            for left in range(cols):
                if input_grid.get_cell(top, left) == color:
                    bottom, right = top, left
                    while bottom + 1 < rows and input_grid.get_cell(bottom + 1, left) == color:
                        bottom += 1
                    while right + 1 < cols and input_grid.get_cell(top, right + 1) == color:
                        right += 1
                    
                    if is_rectangle(color, top, left, bottom, right) and is_isolated(color, top, left, bottom, right):
                        rectangles.append((color, top, left, bottom, right))

    if not rectangles:
        return ColoredGrid([[0]])  # Return a 1x1 grid with black color if no rectangles found

    smallest_rectangle = min(rectangles, key=lambda r: (r[3] - r[1] + 1) * (r[4] - r[2] + 1))
    color, top, left, bottom, right = smallest_rectangle

    result = []
    for r in range(top, bottom + 1):
        row = []
        for c in range(left, right + 1):
            row.append(color)
        result.append(row)

    return ColoredGrid(values=result)
