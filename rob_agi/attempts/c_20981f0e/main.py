from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_20981f0e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Rearrange blue cells (1) in each section between red dot (2) rows to form vertically aligned columns.
    The solution maintains the same number of blue cells in each section, preserves the positions of red dots,
    and attempts to maintain the original distribution and shape of blue cells within each section.
    Steps:
    1. Analyze the input grid and identify sections between red dot rows
    2. For each section, rearrange blue cells to form columns while preserving their original distribution
    3. Ensure at least one empty row between blue cells and red dots where possible
    4. Construct the output grid with the new arrangements
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Find red dot rows and copy them to output grid
    red_dot_rows = []
    for i in range(rows):
        if 2 in input_grid.values[i]:
            red_dot_rows.append(i)
            output_grid.values[i] = input_grid.values[i].copy()
    
    # Define sections
    sections = list(zip([-1] + red_dot_rows, red_dot_rows + [rows]))
    
    # Rearrange blue cells in each section
    for start, end in sections:
        if end - start > 2:
            rearrange_section(input_grid, output_grid, start, end)
    
    return output_grid

def rearrange_section(input_grid: ColoredGrid, output_grid: ColoredGrid, start: int, end: int):
    section_height = end - start - 1
    cols = input_grid.get_dimensions()[1]
    
    # Analyze blue cell distribution
    column_counts = [sum(input_grid.values[r][c] == 1 for r in range(start + 1, end)) for c in range(cols)]
    blue_columns = [c for c, count in enumerate(column_counts) if count > 0]
    
    # Rearrange blue cells
    for c in blue_columns:
        blue_cells = [input_grid.values[r][c] for r in range(start + 1, end)].count(1)
        start_row = max(end - blue_cells - 1, start + 1)
        for r in range(start_row, end):
            output_grid.values[r][c] = 1 if blue_cells > 0 else 0
            blue_cells -= 1

def ensure_empty_rows(grid: ColoredGrid, red_dot_rows: List[int]):
    rows, cols = grid.get_dimensions()
    
    for red_row in red_dot_rows:
        if red_row > 0 and any(grid.values[red_row - 1][c] == 1 for c in range(cols)):
            # Shift blue cells up if possible
            for row in range(red_row - 1, 0, -1):
                if all(grid.values[row - 1][c] == 0 for c in range(cols)):
                    grid.values[row - 1] = grid.values[row]
                    grid.values[row] = [0] * cols
                    break
