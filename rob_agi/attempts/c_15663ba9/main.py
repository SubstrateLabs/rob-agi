from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict
from collections import deque

def solve_15663ba9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on structural features of shapes.
    
    The solution follows these steps:
    1. For each non-black cell:
       - If it's a corner (two adjacent neighbors of same color that are also adjacent), mark it as yellow (4).
       - If it's an intersection (three or four adjacent neighbors of same color), mark it as red (2).
       - If it's an end of a line (only one adjacent neighbor of same color), mark it as yellow (4).
       - Otherwise, keep its original color.
    2. Black cells (0) remain unchanged.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    output_grid = ColoredGrid(values=[[0 for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)])
    
    def is_corner(grid: ColoredGrid, row: int, col: int) -> bool:
        color = grid.get_cell(row, col)
        neighbors = []
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            if 0 <= row+dr < grid.num_rows and 0 <= col+dc < grid.num_cols:
                if grid.get_cell(row+dr, col+dc) == color:
                    neighbors.append((row+dr, col+dc))
        if len(neighbors) == 2:
            r1, c1 = neighbors[0]
            r2, c2 = neighbors[1]
            return abs(r1-r2) + abs(c1-c2) == 1
        return False

    def is_intersection(grid: ColoredGrid, row: int, col: int) -> bool:
        color = grid.get_cell(row, col)
        count = 0
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            if 0 <= row+dr < grid.num_rows and 0 <= col+dc < grid.num_cols:
                if grid.get_cell(row+dr, col+dc) == color:
                    count += 1
        return count >= 3

    def is_end_of_line(grid: ColoredGrid, row: int, col: int) -> bool:
        color = grid.get_cell(row, col)
        count = 0
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            if 0 <= row+dr < grid.num_rows and 0 <= col+dc < grid.num_cols:
                if grid.get_cell(row+dr, col+dc) == color:
                    count += 1
        return count == 1

    for row in range(input_grid.num_rows):
        for col in range(input_grid.num_cols):
            if input_grid.get_cell(row, col) == 0:
                output_grid.set_cell(row, col, 0)
            elif is_corner(input_grid, row, col) or is_end_of_line(input_grid, row, col):
                output_grid.set_cell(row, col, 4)
            elif is_intersection(input_grid, row, col):
                output_grid.set_cell(row, col, 2)
            else:
                output_grid.set_cell(row, col, input_grid.get_cell(row, col))
    
    return output_grid
