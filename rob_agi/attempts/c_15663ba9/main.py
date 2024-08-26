from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict
from collections import deque

def solve_15663ba9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on structural features of shapes.
    
    The solution follows these steps:
    1. For each non-black cell:
       - If it's a corner or endpoint (has 1 or 2 colored neighbors and at least one black diagonal neighbor), mark it as yellow (4).
       - If it's an intersection (four adjacent colored neighbors), mark it as red (2).
       - Otherwise, keep its original color.
    2. Black cells (0) remain unchanged.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    output_grid = ColoredGrid(values=[[0 for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)])
    
    def is_corner_or_endpoint(grid: ColoredGrid, row: int, col: int) -> bool:
        colored_neighbors = count_colored_neighbors(grid, row, col)
        return (colored_neighbors == 1 or colored_neighbors == 2) and has_black_diagonal(grid, row, col)

    def count_colored_neighbors(grid: ColoredGrid, row: int, col: int) -> int:
        count = 0
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            if 0 <= row+dr < grid.num_rows and 0 <= col+dc < grid.num_cols:
                if grid.get_cell(row+dr, col+dc) != 0:
                    count += 1
        return count

    def has_black_diagonal(grid: ColoredGrid, row: int, col: int) -> bool:
        for dr, dc in [(1,1), (1,-1), (-1,1), (-1,-1)]:
            if 0 <= row+dr < grid.num_rows and 0 <= col+dc < grid.num_cols:
                if grid.get_cell(row+dr, col+dc) == 0:
                    return True
        return False

    for row in range(input_grid.num_rows):
        for col in range(input_grid.num_cols):
            cell_value = input_grid.get_cell(row, col)
            if cell_value == 0:
                output_grid.set_cell(row, col, 0)
            else:
                if is_corner_or_endpoint(input_grid, row, col):
                    output_grid.set_cell(row, col, 4)
                elif count_colored_neighbors(input_grid, row, col) == 4:
                    output_grid.set_cell(row, col, 2)
                else:
                    output_grid.set_cell(row, col, cell_value)
    
    return output_grid
