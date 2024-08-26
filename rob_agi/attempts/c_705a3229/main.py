from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_705a3229(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by expanding colored squares into T-shapes or rectangles.
    
    For each colored square:
    1. Determine the longest possible stem direction (up or down).
    2. Create the stem in that direction.
    3. Create a horizontal top at the far end of the stem.
    4. Fill in the resulting shape with the original color.

    The shape will be a T, inverted T, or rectangle, depending on available space.

    Args:
    input_grid (ColoredGrid): The input grid with colored squares.

    Returns:
    ColoredGrid: The transformed grid with expanded shapes.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    def find_colored_squares() -> List[Tuple[int, int, int]]:
        return [(r, c, output_grid.get_cell(r, c)) 
                for r in range(rows) 
                for c in range(cols) 
                if output_grid.get_cell(r, c) != 0]
    
    def count_extension(r: int, c: int, dr: int) -> int:
        count = 0
        while 0 <= r + dr < rows and output_grid.get_cell(r + dr, c) == 0:
            count += 1
            r += dr
        return count
    
    def create_stem(r: int, c: int, color: int, direction: int):
        while 0 <= r < rows and output_grid.get_cell(r, c) == 0:
            output_grid.set_cell(r, c, color)
            r += direction
    
    def create_top(r: int, c: int, color: int):
        for dc in [-1, 1]:
            nc = c
            while 0 <= nc < cols and output_grid.get_cell(r, nc) == 0:
                output_grid.set_cell(r, nc, color)
                nc += dc
    
    for r, c, color in find_colored_squares():
        up_count = count_extension(r, c, -1)
        down_count = count_extension(r, c, 1)
        
        if up_count >= down_count:
            create_stem(r, c, color, -1)
            create_top(r - up_count, c, color)
        else:
            create_stem(r, c, color, 1)
            create_top(r + down_count, c, color)
    
    return output_grid
