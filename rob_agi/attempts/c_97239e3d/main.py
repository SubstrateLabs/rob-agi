from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional

def solve_97239e3d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid expansion challenge by expanding colored squares within their quadrants.
    
    The solution follows these steps:
    1. Define four quadrants in the 16x16 grid: top-left, top-right, bottom-left, bottom-right.
    2. Process the first three quadrants in order: top-left, top-right, bottom-left.
    3. For each quadrant, find the first non-black, non-sky colored square in the quadrant or adjacent quadrants.
    4. If found, fill the quadrant with this color, preserving sky-colored squares.
    5. Repeat the process until no further changes are made, allowing multiple rounds of expansion.
    6. Leave the bottom-right quadrant unchanged.
    7. Preserve the 17th row and column (index 16) from the original grid.
    """
    output_grid = input_grid.deep_copy()
    
    def is_in_quadrant(r: int, c: int, start_row: int, end_row: int, start_col: int, end_col: int) -> bool:
        return start_row <= r < end_row and start_col <= c < end_col

    def find_color_in_quadrant(start_row: int, end_row: int, start_col: int, end_col: int) -> Optional[int]:
        for r in range(start_row, end_row):
            for c in range(start_col, end_col):
                color = output_grid.get_cell(r, c)
                if color not in [0, 8]:
                    return color
        return None

    def find_color_in_adjacent_quadrants(quad_index: int) -> Optional[int]:
        adjacent_quads = {
            0: [1, 2],  # top-left: check top-right and bottom-left
            1: [0, 3],  # top-right: check top-left and bottom-right
            2: [0, 3]   # bottom-left: check top-left and bottom-right
        }
        for adj_quad in adjacent_quads[quad_index]:
            color = find_color_in_quadrant(*quadrants[adj_quad])
            if color is not None:
                return color
        return None

    def expand_color(start_row: int, end_row: int, start_col: int, end_col: int, color: int) -> bool:
        changed = False
        for r in range(start_row, end_row):
            for c in range(start_col, end_col):
                if output_grid.get_cell(r, c) == 0:
                    output_grid.set_cell(r, c, color)
                    changed = True
        return changed

    quadrants = [
        (0, 8, 0, 8),   # top-left
        (0, 8, 8, 16),  # top-right
        (8, 16, 0, 8),  # bottom-left
        (8, 16, 8, 16)  # bottom-right (not processed)
    ]

    changes_made = True
    while changes_made:
        changes_made = False
        for i, quadrant in enumerate(quadrants[:3]):  # Process only the first three quadrants
            color = find_color_in_quadrant(*quadrant)
            if color is None:
                color = find_color_in_adjacent_quadrants(i)
            if color is not None:
                changes_made |= expand_color(*quadrant, color)

    # Preserve the 17th row and column
    for i in range(17):
        output_grid.set_cell(16, i, input_grid.get_cell(16, i))
        output_grid.set_cell(i, 16, input_grid.get_cell(i, 16))
    
    return output_grid
