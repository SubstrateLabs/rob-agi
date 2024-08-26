from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import math

def solve_c64f1187(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by:
    1. Processing the upper section to create 2x2 colored rectangles from non-blue, non-zero colors.
    2. Adding a separator row of zeros.
    3. Processing the lower section to create 2x2 colored rectangles with single-cell extensions
       below the left cell for all but the rightmost rectangle in each row.
    4. Arranging these processed elements in a compact output grid.
    5. Optimizing the output grid by removing trailing empty rows and columns.
    """
    upper_colors = process_upper_section(input_grid)
    lower_rows = process_lower_section(input_grid)
    
    width = max(len(upper_colors) * 3 - 1, max((len(row) * 3 - 1 for row in lower_rows), default=0))
    height = math.ceil(len(upper_colors) / 2) * 2 + 1 + len(lower_rows) * 2
    
    output_grid = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])
    
    # Place upper section rectangles
    for i, color in enumerate(upper_colors):
        row = (i // 2) * 2
        col = (i % 2) * 3
        place_2x2_rectangle(output_grid, color, row, col)
        if i % 2 == 0 and i < len(upper_colors) - 1:
            output_grid.values[row + 1][col + 1] = 0
    
    # Add separator row
    separator_row = math.ceil(len(upper_colors) / 2) * 2
    
    # Place lower section rectangles
    lower_start = separator_row + 1
    for row_index, row in enumerate(lower_rows):
        for col_index, color in enumerate(row):
            place_2x2_rectangle(output_grid, color, lower_start + row_index * 2, col_index * 3)
            if col_index < len(row) - 1:  # Add extension for all but the rightmost rectangle
                output_grid.values[lower_start + row_index * 2 + 1][col_index * 3] = color
    
    # Optimize the output grid
    output_grid = remove_trailing_empty_rows_and_columns(output_grid)
    
    return output_grid

def process_upper_section(grid: ColoredGrid) -> List[int]:
    colors = []
    for row in grid.values:
        if 1 in row:  # Stop when we find a row with blue (1)
            break
        colors.extend([cell for cell in row if cell not in [0, 1]])
    return colors

def process_lower_section(grid: ColoredGrid) -> List[List[int]]:
    rows = []
    start_row = next((i for i, row in enumerate(grid.values) if 1 in row), 0) + 1
    for row_group in range(start_row, len(grid.values), 3):
        if row_group + 2 >= len(grid.values):
            break
        colors = []
        for col in range(0, len(grid.values[0]), 3):
            if col + 2 >= len(grid.values[0]):
                break
            center_color = grid.values[row_group + 1][col + 1]
            if center_color not in [0, 5]:
                colors.append(center_color)
        if colors:
            rows.append(colors)
    return rows

def place_2x2_rectangle(grid: ColoredGrid, color: int, row: int, col: int):
    for i in range(2):
        for j in range(2):
            if row + i < len(grid.values) and col + j < len(grid.values[0]):
                grid.values[row + i][col + j] = color

def remove_trailing_empty_rows_and_columns(grid: ColoredGrid) -> ColoredGrid:
    # Remove trailing empty rows
    while grid.values and all(cell == 0 for cell in grid.values[-1]):
        grid.values.pop()
    
    # Remove trailing empty columns
    if grid.values:
        width = len(grid.values[0])
        while width > 0 and all(row[width-1] == 0 for row in grid.values):
            for row in grid.values:
                row.pop()
            width -= 1
    
    return grid
