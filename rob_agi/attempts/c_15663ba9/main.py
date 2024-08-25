from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict
from collections import deque

def solve_15663ba9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on local cell neighborhoods.
    
    The solution follows these steps:
    1. For each non-black cell:
       - If it's part of a 2x2 square of the same color, mark it as red (2).
       - If it has 2 or more orthogonally adjacent cells of the same color, mark it as red (2).
       - Otherwise, mark it as yellow (4).
    2. Black cells (0) remain unchanged.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    output_grid = ColoredGrid(values=[[0 for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)])
    
    def is_part_of_2x2_square(grid: ColoredGrid, row: int, col: int) -> bool:
        color = grid.get_cell(row, col)
        if row + 1 < grid.num_rows and col + 1 < grid.num_cols:
            return all(grid.get_cell(row+dr, col+dc) == color for dr, dc in [(0,0), (0,1), (1,0), (1,1)])
        return False

    def count_same_color_neighbors(grid: ColoredGrid, row: int, col: int) -> int:
        color = grid.get_cell(row, col)
        count = 0
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            if 0 <= row+dr < grid.num_rows and 0 <= col+dc < grid.num_cols:
                if grid.get_cell(row+dr, col+dc) == color:
                    count += 1
        return count

    for row in range(input_grid.num_rows):
        for col in range(input_grid.num_cols):
            if input_grid.get_cell(row, col) == 0:
                output_grid.set_cell(row, col, 0)
            elif is_part_of_2x2_square(input_grid, row, col):
                output_grid.set_cell(row, col, 2)
            elif count_same_color_neighbors(input_grid, row, col) >= 2:
                output_grid.set_cell(row, col, 2)
            else:
                output_grid.set_cell(row, col, 4)
    
    return output_grid
