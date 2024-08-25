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
    standardizing them to 4x4 size, and arranging them in a new grid.
    The function ignores solid squares and focuses only on outlines.
    Outlines are sorted by their position in the input grid (top-to-bottom, left-to-right)
    and arranged horizontally if there are three or more, vertically if there are two.
    """
    outlines: List[Tuple[int, int, int, int]] = []  # (color, row, col, size)
    standard_size = 4

    # Scan the input grid for square outlines
    for row in range(input_grid.num_rows):
        for col in range(input_grid.num_cols):
            if input_grid.get_cell(row, col) != 0:
                is_outline, size = is_square_outline(input_grid, row, col)
                if is_outline:
                    outlines.append((input_grid.get_cell(row, col), row, col, size))

    if not outlines:
        return ColoredGrid(values=[[0]])

    # Sort outlines by position (top-to-bottom, left-to-right)
    outlines.sort(key=lambda x: (x[1], x[2]))

    # Determine arrangement
    if len(outlines) == 1:
        output_height = 4
        output_width = 4
    elif len(outlines) == 2:
        output_height = 8
        output_width = 4
    else:
        output_height = 4
        output_width = 4 * len(outlines)

    # Create output grid
    result = ColoredGrid(values=[[0 for _ in range(output_width)] for _ in range(output_height)])

    # Place standardized outlines in the output grid
    for i, (color, src_row, src_col, _) in enumerate(outlines):
        if len(outlines) == 2:
            tgt_row = i * 4
            tgt_col = 0
        else:
            tgt_row = 0
            tgt_col = i * 4
        standardized = standardize_outline(input_grid, (color, src_row, src_col, standard_size), standard_size)
        copy_subgrid(standardized, result, 0, 0, tgt_row, tgt_col, standard_size, standard_size)

    return result

def standardize_outline(grid: ColoredGrid, outline: Tuple[int, int, int, int], standard_size: int) -> ColoredGrid:
    """Create a standardized 4x4 outline from the given outline."""
    color, row, col, _ = outline
    result = ColoredGrid(values=[[0 for _ in range(standard_size)] for _ in range(standard_size)])
    for i in range(standard_size):
        result.set_cell(0, i, color)
        result.set_cell(standard_size - 1, i, color)
        result.set_cell(i, 0, color)
        result.set_cell(i, standard_size - 1, color)
    return result

def copy_subgrid(source: ColoredGrid, target: ColoredGrid, src_row: int, src_col: int, 
                 tgt_row: int, tgt_col: int, height: int, width: int):
    """Copy a subgrid from the source to the target grid."""
    for i in range(height):
        for j in range(width):
            value = source.get_cell(src_row + i, src_col + j)
            target.set_cell(tgt_row + i, tgt_col + j, value)
