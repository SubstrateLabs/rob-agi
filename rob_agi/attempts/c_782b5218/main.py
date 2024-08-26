from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_782b5218(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on color distribution and pattern detection.
    
    1. Identifies unique colors in the input grid.
    2. Determines if the pattern should be horizontal banding or diagonal based on the number of unique colors and middle row uniformity.
    3. For horizontal banding (only if there are exactly 3 unique colors and the middle row is uniform): 
       - Divides the grid into three horizontal bands.
       - Fills the top band with the lowest color, preserves the middle row, and fills the bottom band with the highest color.
    4. For diagonal pattern (in all other cases): 
       - Fills diagonally from top-left to bottom-right with sorted colors.
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    rows, cols = input_grid.get_dimensions()
    unique_colors = sorted(set(color for row in input_grid.values for color in row))
    
    if len(unique_colors) == 3 and has_uniform_middle_row(input_grid):
        return create_horizontal_banding(input_grid, unique_colors)
    else:
        return create_diagonal_pattern(input_grid, unique_colors)

def has_uniform_middle_row(grid: ColoredGrid) -> bool:
    middle_row = len(grid.values) // 2
    return len(set(grid.values[middle_row])) == 1

def create_horizontal_banding(grid: ColoredGrid, sorted_colors: List[int]) -> ColoredGrid:
    rows, cols = grid.get_dimensions()
    new_values = [[0 for _ in range(cols)] for _ in range(rows)]
    
    middle_row = rows // 2
    band_height = (rows - 1) // 3
    
    # Fill top band
    for r in range(middle_row - band_height):
        new_values[r] = [sorted_colors[0]] * cols
    
    # Preserve middle row
    new_values[middle_row] = grid.values[middle_row]
    
    # Fill bottom band
    for r in range(middle_row + 1, rows):
        new_values[r] = [sorted_colors[2]] * cols
    
    return ColoredGrid(values=new_values)

def create_diagonal_pattern(grid: ColoredGrid, sorted_colors: List[int]) -> ColoredGrid:
    rows, cols = grid.get_dimensions()
    new_values = [[0 for _ in range(cols)] for _ in range(rows)]
    num_colors = len(sorted_colors)
    diagonal_width = (rows + cols) // num_colors
    
    for r in range(rows):
        for c in range(cols):
            diagonal_index = (r + c) // diagonal_width
            color_index = min(diagonal_index, num_colors - 1)
            new_values[r][c] = sorted_colors[color_index]
    
    return ColoredGrid(values=new_values)
