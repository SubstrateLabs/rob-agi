from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_97239e3d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid expansion challenge by expanding colored squares within their quadrants.
    
    The solution follows these steps:
    1. Scan the input grid to identify non-black and non-sky colored squares.
    2. Create an expansion order based on top-to-bottom, left-to-right position.
    3. For each colored square, expand horizontally and vertically within its quadrant.
    4. Preserve the 17th row and column (index 16) from the original grid.
    5. Handle multiple colors in the same quadrant by processing them in order.
    
    Expansion stops at quadrant boundaries, sky (8) squares, or other expanded colors.
    """
    output_grid = input_grid.deep_copy()
    
    def get_colored_squares() -> List[Tuple[int, int, int]]:
        return [(r, c, input_grid.get_cell(r, c)) 
                for r in range(17) 
                for c in range(17) 
                if input_grid.get_cell(r, c) not in [0, 8]]
    
    colored_squares = sorted(get_colored_squares(), key=lambda x: (x[0], x[1]))
    
    def expand_in_quadrant(row: int, col: int, color: int):
        row_start = 0 if row < 8 else 9
        row_end = 7 if row < 8 else 16
        col_start = 0 if col < 8 else 9
        col_end = 7 if col < 8 else 16

        # Expand horizontally
        for c in range(col_start, col_end + 1):
            if output_grid.get_cell(row, c) in [0, color]:
                output_grid.set_cell(row, c, color)

        # Expand vertically
        for r in range(row_start, row_end + 1):
            if output_grid.get_cell(r, col) in [0, color]:
                output_grid.set_cell(r, col, color)

        # Ensure the original position is colored (intersection point)
        output_grid.set_cell(row, col, color)
    
    for row, col, color in colored_squares:
        if row != 16 and col != 16:  # Skip expansion for 17th row/column
            expand_in_quadrant(row, col, color)
    
    return output_grid
