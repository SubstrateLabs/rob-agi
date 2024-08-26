from rob_agi.colored_grid import ColoredGrid
from typing import List

def solve_782b5218(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a diagonal pattern of triangular color regions.
    
    1. Identifies unique colors in the input grid.
    2. Sorts the colors in ascending order.
    3. Creates a diagonal pattern from top-left to bottom-right with sorted colors.
    4. Each color forms a triangular region, with the lowest color in the top-left
       and the highest color in the bottom-right.
    5. The red color (2) always forms a diagonal strip separating other colors.
    
    Returns a new ColoredGrid with the transformed diagonal pattern of triangular regions.
    """
    rows, cols = input_grid.get_dimensions()
    unique_colors = sorted(set(color for row in input_grid.values for color in row))
    return create_diagonal_triangles(rows, cols, unique_colors)

def create_diagonal_triangles(rows: int, cols: int, sorted_colors: List[int]) -> ColoredGrid:
    new_values = [[0 for _ in range(cols)] for _ in range(rows)]
    num_colors = len(sorted_colors)
    
    for r in range(rows):
        for c in range(cols):
            diagonal_index = r + c
            color_index = min(diagonal_index // 2, num_colors - 1)
            new_values[r][c] = sorted_colors[color_index]
    
    return ColoredGrid(values=new_values)
