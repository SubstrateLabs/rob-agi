from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_97239e3d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid expansion challenge by expanding colored squares within their quadrants.
    
    The solution follows these steps:
    1. Scan the input grid to identify non-black and non-sky colored squares.
    2. Sort colored squares based on their position (top-to-bottom, left-to-right).
    3. For each colored square, calculate the maximum possible expansion within its quadrant.
    4. Fill the expanded area with the color, respecting sky-colored squares.
    5. Preserve the 17th row and column (index 16) from the original grid.
    
    Expansion is limited by quadrant boundaries, sky (8) squares, or other expanded colors.
    """
    output_grid = input_grid.deep_copy()
    
    def get_colored_squares() -> List[Tuple[int, int, int]]:
        return [(r, c, input_grid.get_cell(r, c)) 
                for r in range(16) 
                for c in range(16) 
                if input_grid.get_cell(r, c) not in [0, 8]]
    
    colored_squares = sorted(get_colored_squares(), key=lambda x: (x[0], x[1]))
    
    def expand_in_quadrant(row: int, col: int, color: int):
        # Calculate maximum expansion in each direction
        up = min(row, 7 - row) if row < 8 else min(row - 8, 7)
        down = min(7 - row, row) if row < 8 else min(15 - row, row - 8)
        left = min(col, 7 - col) if col < 8 else min(col - 8, 7)
        right = min(7 - col, col) if col < 8 else min(15 - col, col - 8)
        
        expansion_size = min(up, down, left, right)
        
        start_x = row - expansion_size if row >= 8 else row
        start_y = col - expansion_size if col >= 8 else col
        
        for i in range(start_x, start_x + expansion_size * 2 + 1):
            for j in range(start_y, start_y + expansion_size * 2 + 1):
                if output_grid.get_cell(i, j) != 8:  # Preserve sky-colored squares
                    output_grid.set_cell(i, j, color)
    
    for row, col, color in colored_squares:
        expand_in_quadrant(row, col, color)
    
    # Preserve the 17th row and column
    for i in range(17):
        output_grid.set_cell(16, i, input_grid.get_cell(16, i))
        output_grid.set_cell(i, 16, input_grid.get_cell(i, 16))
    
    return output_grid
