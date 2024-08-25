from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_20981f0e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Rearrange blue cells (1) in each section between red dot (2) rows to form vertically aligned columns.
    The solution maintains the same number of blue cells in each section and preserves the positions of red dots.
    Steps:
    1. Analyze the input grid and identify sections between red dot rows
    2. Identify global column positions for blue cells across all sections
    3. For each section, distribute blue cells among the global columns, prioritizing original positions
    4. Align columns vertically across all sections
    5. Ensure at least one empty row between blue cells and red dots
    6. Construct the output grid with the new arrangements
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Find red dot rows and copy them to output grid
    red_dot_rows = []
    for i in range(rows):
        if 2 in input_grid.values[i]:
            red_dot_rows.append(i)
            output_grid.values[i] = input_grid.values[i].copy()
    
    # Define sections and analyze blue cell positions
    sections = list(zip([-1] + red_dot_rows, red_dot_rows + [rows]))
    global_columns = set()
    section_columns = {}
    
    for start, end in sections:
        if end - start > 2:
            columns = defaultdict(int)
            for r in range(start + 1, end):
                for c in range(cols):
                    if input_grid.values[r][c] == 1:
                        columns[c] += 1
                        global_columns.add(c)
            section_columns[(start, end)] = columns
    
    global_columns = sorted(list(global_columns))
    
    # Rearrange blue cells in each section
    for start, end in sections:
        if end - start > 2:
            rearrange_section(input_grid, output_grid, start, end, global_columns, section_columns[(start, end)])
    
    align_columns_vertically(output_grid, sections, global_columns)
    ensure_empty_rows(output_grid, red_dot_rows)
    
    return output_grid

def rearrange_section(input_grid: ColoredGrid, output_grid: ColoredGrid, start: int, end: int, global_columns: List[int], section_columns: Dict[int, int]):
    total_blue_cells = sum(section_columns.values())
    cells_per_column = total_blue_cells // len(global_columns)
    extra_cells = total_blue_cells % len(global_columns)
    
    column_assignment = {col: cells_per_column + (1 if i < extra_cells else 0) for i, col in enumerate(global_columns)}
    
    # Prioritize original positions
    for col in sorted(section_columns, key=section_columns.get, reverse=True):
        if col in global_columns:
            assigned = min(section_columns[col], column_assignment[col])
            column_assignment[col] -= assigned
            total_blue_cells -= assigned
    
    # Distribute remaining cells
    for col in global_columns:
        while column_assignment[col] > 0 and total_blue_cells > 0:
            column_assignment[col] -= 1
            total_blue_cells -= 1
    
    # Place blue cells in the output grid
    for col, count in column_assignment.items():
        for i in range(count):
            row = end - 1 - i
            output_grid.values[row][col] = 1

def align_columns_vertically(grid: ColoredGrid, sections: List[Tuple[int, int]], global_columns: List[int]):
    rows, _ = grid.get_dimensions()
    max_heights = {col: 0 for col in global_columns}
    
    for start, end in sections:
        if end - start > 2:
            for col in global_columns:
                height = sum(grid.values[r][col] for r in range(start + 1, end))
                max_heights[col] = max(max_heights[col], height)
    
    for start, end in sections:
        if end - start > 2:
            for col in global_columns:
                blue_cells = [r for r in range(start + 1, end) if grid.values[r][col] == 1]
                offset = max_heights[col] - len(blue_cells)
                for i, row in enumerate(blue_cells):
                    grid.values[row][col] = 0
                    grid.values[end - offset - i - 1][col] = 1

def ensure_empty_rows(grid: ColoredGrid, red_dot_rows: List[int]):
    rows, cols = grid.get_dimensions()
    
    for red_row in red_dot_rows:
        if red_row > 0 and any(grid.values[red_row - 1][c] == 1 for c in range(cols)):
            # Shift blue cells up
            for row in range(1, red_row):
                for col in range(cols):
                    grid.values[row - 1][col] = grid.values[row][col]
            grid.values[red_row - 1] = [0] * cols
