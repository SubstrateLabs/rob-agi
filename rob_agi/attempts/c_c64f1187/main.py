from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_c64f1187(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by:
    1. Processing the upper section (first 3 rows) to create 2x2 colored rectangles.
    2. Processing the lower section (remaining rows) to create 2x2 colored rectangles
       with single-cell extensions below the left cell for all but the rightmost rectangle.
    3. Arranging these processed elements in a compact output grid.
    """
    upper_colors = process_upper_section(input_grid)
    lower_rows = process_lower_section(input_grid)
    
    width = max(len(upper_colors) * 2, max(len(row) for row in lower_rows) * 2)
    height = len(upper_colors) * 2 + 1 + len(lower_rows) * 2
    
    output_grid = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])
    
    # Place upper section rectangles
    for i, color in enumerate(upper_colors):
        place_2x2_rectangle(output_grid, color, 0, i * 2)
    
    # Place lower section rectangles
    for row_index, row in enumerate(lower_rows):
        for col_index, color in enumerate(row):
            place_2x2_rectangle(output_grid, color, len(upper_colors) * 2 + 1 + row_index * 2, col_index * 2)
            if col_index < len(row) - 1:  # Add extension for all but the rightmost rectangle
                output_grid.values[len(upper_colors) * 2 + 2 + row_index * 2][col_index * 2] = color
    
    return output_grid

def process_upper_section(grid: ColoredGrid) -> List[int]:
    return [cell for row in grid.values[:3] for cell in row if cell != 0]

def process_lower_section(grid: ColoredGrid) -> List[List[int]]:
    rows = []
    for row_group in range(3, len(grid.values), 3):
        colors = []
        for col in range(0, len(grid.values[0]), 3):
            center_color = grid.values[row_group + 1][col + 1]
            if center_color != 0 and center_color != 5:
                colors.append(center_color)
        if colors:
            rows.append(colors)
    return rows

def place_2x2_rectangle(grid: ColoredGrid, color: int, row: int, col: int):
    for i in range(2):
        for j in range(2):
            grid.values[row + i][col + j] = color
