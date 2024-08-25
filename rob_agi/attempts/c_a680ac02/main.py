from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import math

def is_square_outline(grid: ColoredGrid, row: int, col: int) -> Tuple[bool, int]:
    """Check if the cell at (row, col) is part of a square outline."""
    color = grid.get_cell(row, col)
    size = 1
    while (row + size < grid.num_rows and col + size < grid.num_cols and
           grid.get_cell(row + size, col) == color and
           grid.get_cell(row, col + size) == color):
        size += 1
    
    if size < 2:
        return False, 0
    
    for i in range(size):
        if (grid.get_cell(row + i, col) != color or
            grid.get_cell(row + i, col + size - 1) != color or
            grid.get_cell(row, col + i) != color or
            grid.get_cell(row + size - 1, col + i) != color):
            return False, 0
    
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            if grid.get_cell(row + i, col + j) != 0:
                return False, 0
    
    return True, size

def copy_outline(source: ColoredGrid, target: ColoredGrid, src_row: int, src_col: int, tgt_row: int, tgt_col: int, size: int):
    """Copy a square outline from the source grid to the target grid."""
    color = source.get_cell(src_row, src_col)
    for i in range(size):
        target.set_cell(tgt_row, tgt_col + i, color)
        target.set_cell(tgt_row + size - 1, tgt_col + i, color)
        target.set_cell(tgt_row + i, tgt_col, color)
        target.set_cell(tgt_row + i, tgt_col + size - 1, color)

def solve_a680ac02(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the challenge by identifying square outlines in the input grid,
    arranging them horizontally or vertically based on the optimal layout,
    and returning a new grid with the arranged outlines.
    """
    outlines: List[Tuple[int, int, int, int]] = []  # (color, row, col, size)
    
    for row in range(input_grid.num_rows):
        for col in range(input_grid.num_cols):
            if input_grid.get_cell(row, col) != 0:
                is_outline, size = is_square_outline(input_grid, row, col)
                if is_outline:
                    outlines.append((input_grid.get_cell(row, col), row, col, size))
    
    if not outlines:
        return ColoredGrid(values=[[]])
    
    if len(outlines) == 1:
        color, row, col, size = outlines[0]
        return input_grid.extract_subgrid(row, col, size, size)
    
    total_area = sum(size * size for _, _, _, size in outlines)
    ideal_side = math.isqrt(total_area)
    
    if len(outlines) <= ideal_side:
        # Arrange horizontally
        height = outlines[0][3]
        width = sum(size for _, _, _, size in outlines)
    else:
        # Arrange vertically
        height = sum(size for _, _, _, size in outlines)
        width = outlines[0][3]
    
    result = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])
    
    current_row, current_col = 0, 0
    for color, src_row, src_col, size in outlines:
        copy_outline(input_grid, result, src_row, src_col, current_row, current_col, size)
        if len(outlines) <= ideal_side:
            current_col += size
        else:
            current_row += size
    
    return result
