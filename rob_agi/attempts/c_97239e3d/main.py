from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional

def solve_97239e3d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid expansion challenge by expanding colored squares within their quadrants.
    
    The solution follows these steps:
    1. Define four quadrants in the 16x16 grid.
    2. For each quadrant, find the non-black, non-sky colored square closest to its corner.
    3. Fill the entire quadrant with the color of the closest square.
    4. Process quadrants in order: top-left, top-right, bottom-left, bottom-right.
    5. Preserve the 17th row and column (index 16) from the original grid.
    
    Expansion fills the entire quadrant, overwriting existing colors except in the 17th row and column.
    """
    output_grid = input_grid.deep_copy()
    
    def get_quadrant(row: int, col: int) -> str:
        if row < 8:
            return "top-left" if col < 8 else "top-right"
        else:
            return "bottom-left" if col < 8 else "bottom-right"
    
    def find_closest_color(quadrant: str) -> Optional[int]:
        start_row, end_row = (0, 8) if "top" in quadrant else (8, 16)
        start_col, end_col = (0, 8) if "left" in quadrant else (8, 16)
        closest_color = None
        min_distance = float('inf')
        
        for r in range(start_row, end_row):
            for c in range(start_col, end_col):
                color = input_grid.get_cell(r, c)
                if color not in [0, 8]:
                    if "top" in quadrant:
                        distance = r + (c if "left" in quadrant else 15 - c)
                    else:
                        distance = (15 - r) + (c if "left" in quadrant else 15 - c)
                    if distance < min_distance:
                        min_distance = distance
                        closest_color = color
        
        return closest_color
    
    def fill_quadrant(quadrant: str, color: int):
        start_row, end_row = (0, 8) if "top" in quadrant else (8, 16)
        start_col, end_col = (0, 8) if "left" in quadrant else (8, 16)
        for r in range(start_row, end_row):
            for c in range(start_col, end_col):
                output_grid.set_cell(r, c, color)
    
    for quadrant in ["top-left", "top-right", "bottom-left", "bottom-right"]:
        color = find_closest_color(quadrant)
        if color is not None:
            fill_quadrant(quadrant, color)
    
    # Preserve the 17th row and column
    for i in range(17):
        output_grid.set_cell(16, i, input_grid.get_cell(16, i))
        output_grid.set_cell(i, 16, input_grid.get_cell(i, 16))
    
    return output_grid
