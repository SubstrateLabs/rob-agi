from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import math

def solve_20981f0e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Rearrange blue cells (1) in each section between red dot (2) rows to form two vertically aligned columns.
    The solution maintains the same number of blue cells in each section and preserves the positions of red dots.
    Steps:
    1. Identify sections between red dot rows
    2. For each section, count blue cells and distribute them into two columns
    3. Align columns vertically across all sections
    4. Ensure at least one empty row between blue cells and red dots
    5. Construct the output grid with the new arrangements
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Find red dot rows and copy them to output grid
    red_dot_rows = []
    for i in range(rows):
        if 2 in input_grid.values[i]:
            red_dot_rows.append(i)
            output_grid.values[i] = input_grid.values[i].copy()
    
    # Process each section between red dot rows
    sections = list(zip([-1] + red_dot_rows, red_dot_rows + [rows]))
    for start, end in sections:
        if end - start > 2:  # Ensure space for blue cells and empty row
            rearrange_section(input_grid, output_grid, start + 1, end - 1, cols)
    
    align_columns_vertically(output_grid, sections)
    ensure_empty_rows(output_grid, red_dot_rows)
    
    return output_grid

def rearrange_section(input_grid: ColoredGrid, output_grid: ColoredGrid, start: int, end: int, cols: int):
    blue_cells = sum(input_grid.values[r].count(1) for r in range(start, end + 1))
    left_count = math.ceil(blue_cells / 2)
    right_count = blue_cells - left_count
    
    center = cols // 2
    left_col, right_col = center - 1, center
    
    for i in range(left_count):
        row = end - i
        output_grid.values[row][left_col] = 1
    
    for i in range(right_count):
        row = end - i
        output_grid.values[row][right_col] = 1

def align_columns_vertically(grid: ColoredGrid, sections: List[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    center = cols // 2
    left_col, right_col = center - 1, center
    
    max_height = max(sum(grid.values[start+1:end].count(1) for start, end in sections if end - start > 2))
    
    for start, end in sections:
        if end - start > 2:
            section_height = sum(grid.values[start+1:end].count(1))
            offset = max_height - section_height
            
            # Move blue cells up
            for row in range(start + 1, end):
                if grid.values[row][left_col] == 1:
                    grid.values[row - offset][left_col] = 1
                    grid.values[row][left_col] = 0
                if grid.values[row][right_col] == 1:
                    grid.values[row - offset][right_col] = 1
                    grid.values[row][right_col] = 0

def ensure_empty_rows(grid: ColoredGrid, red_dot_rows: List[int]):
    rows, cols = grid.get_dimensions()
    center = cols // 2
    left_col, right_col = center - 1, center
    
    for red_row in red_dot_rows:
        if red_row > 0 and (grid.values[red_row - 1][left_col] == 1 or grid.values[red_row - 1][right_col] == 1):
            # Shift blue cells up
            for row in range(1, red_row):
                grid.values[row - 1][left_col] = grid.values[row][left_col]
                grid.values[row - 1][right_col] = grid.values[row][right_col]
            grid.values[red_row - 1][left_col] = 0
            grid.values[red_row - 1][right_col] = 0
