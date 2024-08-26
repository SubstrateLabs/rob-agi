from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional

def solve_97239e3d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid expansion challenge by expanding colored squares within their quadrants.
    
    The solution follows these steps:
    1. Define four quadrants in the 16x16 grid: top-left, top-right, bottom-left, bottom-right.
    2. Process the first three quadrants in order: top-left, top-right, bottom-left.
    3. For each quadrant, find the first non-black, non-sky colored square.
    4. If found, fill the quadrant with this color, preserving sky-colored squares.
    5. Process all three quadrants, allowing multiple expansions.
    6. Leave the bottom-right quadrant unchanged.
    7. Preserve the 17th row and column (index 16) from the original grid.
    """
    output_grid = input_grid.deep_copy()
    
    def process_quadrant(start_row: int, end_row: int, start_col: int, end_col: int):
        for r in range(start_row, end_row):
            for c in range(start_col, end_col):
                color = input_grid.get_cell(r, c)
                if color not in [0, 8]:
                    for rr in range(start_row, end_row):
                        for cc in range(start_col, end_col):
                            if output_grid.get_cell(rr, cc) != 8:
                                output_grid.set_cell(rr, cc, color)
                    return

    quadrants = [
        (0, 8, 0, 8),   # top-left
        (0, 8, 8, 16),  # top-right
        (8, 16, 0, 8)   # bottom-left
    ]

    for quadrant in quadrants:
        process_quadrant(*quadrant)

    # Preserve the 17th row and column
    for i in range(17):
        output_grid.set_cell(16, i, input_grid.get_cell(16, i))
        output_grid.set_cell(i, 16, input_grid.get_cell(i, 16))
    
    return output_grid
