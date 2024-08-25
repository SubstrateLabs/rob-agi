from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_97239e3d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid expansion challenge by expanding colored squares horizontally and vertically.
    
    The solution follows these steps:
    1. Scan the input grid to identify non-black and non-sky colored squares.
    2. Create an expansion order based on left-to-right, top-to-bottom position.
    3. Perform horizontal expansion for each colored square.
    4. Perform vertical expansion for each horizontally expanded region.
    5. Resolve conflicts based on the original expansion order.
    
    Expansion stops at grid edges, sky (8) squares, or other expanded colors.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    
    def get_colored_squares() -> List[Tuple[int, int, int]]:
        return [(r, c, input_grid.get_cell(r, c)) 
                for r in range(rows) 
                for c in range(cols) 
                if input_grid.get_cell(r, c) not in [0, 8]]
    
    colored_squares = sorted(get_colored_squares(), key=lambda x: (x[1], x[0]))
    
    def expand_horizontally(r: int, c: int, color: int):
        left = c
        while left > 0 and output_grid.get_cell(r, left - 1) in [0, color]:
            left -= 1
            output_grid.set_cell(r, left, color)
        
        right = c
        while right < cols - 1 and output_grid.get_cell(r, right + 1) in [0, color]:
            right += 1
            output_grid.set_cell(r, right, color)
        
        return left, right
    
    def expand_vertically(r: int, c: int, color: int):
        top = r
        while top > 0 and output_grid.get_cell(top - 1, c) in [0, color]:
            top -= 1
            output_grid.set_cell(top, c, color)
        
        bottom = r
        while bottom < rows - 1 and output_grid.get_cell(bottom + 1, c) in [0, color]:
            bottom += 1
            output_grid.set_cell(bottom, c, color)
    
    for r, c, color in colored_squares:
        left, right = expand_horizontally(r, c, color)
        for col in range(left, right + 1):
            expand_vertically(r, col, color)
    
    return output_grid
